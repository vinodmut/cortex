package evolve

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// Evolve is the plugin's view of an altk-evolve service. The concrete
// implementation (mcpEvolve) calls the stock evolve MCP server's tools; tests
// inject a fake.
type Evolve interface {
	// Guidelines returns the guidelines relevant to a task from a namespace.
	Guidelines(ctx context.Context, namespace, task, sessionID string) ([]Guideline, error)
	// Learn sends a trajectory batch to evolve, which generates guidelines from
	// it into namespace.
	Learn(ctx context.Context, namespace string, t Trajectory) error
}

// Guideline is one served guideline plus the provenance agentop shows for it.
type Guideline struct {
	ID       string
	Content  string
	Revision *int
	// Source is the trajectory the guideline was learned from (source_task_id).
	Source string
}

// Trajectory is evolve's processing input (altk_evolve.processing.Trajectory).
type Trajectory struct {
	Messages        []map[string]any `json:"messages"`
	ContextMessages []map[string]any `json:"context_messages,omitempty"`
	Batch           *Batch           `json:"batch,omitempty"`
	TraceID         string           `json:"trace_id,omitempty"`
	Model           string           `json:"model,omitempty"`
	Metadata        map[string]any   `json:"metadata,omitempty"`
}

// Batch is evolve's idempotency key for one bounded contribution to a
// conversation: re-sending the same batch is skipped by evolve.
type Batch struct {
	Source         string `json:"source"`
	ConversationID string `json:"conversation_id"`
	BatchID        string `json:"batch_id"`
	Scope          string `json:"scope,omitempty"`
}

// ErrUnavailable wraps every failure to get an answer from evolve: transport
// errors, timeouts, tool errors and undecodable results. The plugin fails open
// on all of them.
var ErrUnavailable = errors.New("evolve unavailable")

// maxServedGuidelines is the most guidelines read from the served namespace
// (list_entities caps a page at 200).
const maxServedGuidelines = 200

// sentinelHeader marks the plugin's own calls so that, should one ever loop
// back through the listener, OnRequest skips it instead of recursing.
const sentinelHeader = "X-Evolve-Cortex"

// profileDefinition is the processing profile the plugin owns on the evolve
// server: standard guideline generation. ensure_processing_profile creates it
// once and keeps operator edits made to it afterwards.
var profileDefinition = map[string]any{
	"processors": []any{map[string]any{
		"id": "guidelines", "plugin": "evolve.guidelines",
		"config": map[string]any{"guidelines_mode": "standard"},
	}},
}

// mcpEvolve talks to `evolve-mcp --transport sse`. One SSE session is kept
// open and re-established after a failure.
//
// Its http.Client never uses a proxy: Cortex is usually also the machine's
// HTTPS_PROXY, and routing evolve's calls through it would record them as
// agent traffic. The sentinel header is belt-and-suspenders on top of that.
type mcpEvolve struct {
	url     string
	profile string
	timeout time.Duration
	http    *http.Client

	mu      sync.Mutex
	client  *mcpclient.Client
	ensured bool
}

func newMCPEvolve(url, profile string, timeout time.Duration) *mcpEvolve {
	return &mcpEvolve{
		url:     url,
		profile: profile,
		timeout: timeout,
		// No client timeout: the SSE stream is long-lived. Calls are bounded by ctx.
		http: &http.Client{Transport: &http.Transport{Proxy: nil, MaxIdleConnsPerHost: 4, IdleConnTimeout: 90 * time.Second}},
	}
}

func (e *mcpEvolve) session(ctx context.Context) (*mcpclient.Client, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.client != nil {
		return e.client, nil
	}
	c, err := mcpclient.NewSSEMCPClient(e.url,
		transport.WithHTTPClient(e.http),
		transport.WithHeaders(map[string]string{sentinelHeader: "1"}))
	if err != nil {
		return nil, err
	}
	// The stream outlives this call, so it must not inherit the caller's deadline.
	if err := c.Start(context.Background()); err != nil {
		return nil, err
	}
	init := mcp.InitializeRequest{}
	init.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	init.Params.ClientInfo = mcp.Implementation{Name: "cortex-evolve-plugin", Version: "1"}
	if _, err := c.Initialize(ctx, init); err != nil {
		_ = c.Close()
		return nil, err
	}
	e.client = c
	return c, nil
}

func (e *mcpEvolve) drop(c *mcpclient.Client) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.client == c {
		_ = c.Close()
		e.client = nil
		e.ensured = false
	}
}

// call invokes a tool and returns its text result.
func (e *mcpEvolve) call(ctx context.Context, tool string, args map[string]any) (string, error) {
	c, err := e.session(ctx)
	if err != nil {
		return "", fmt.Errorf("%w: connect: %v", ErrUnavailable, err)
	}
	req := mcp.CallToolRequest{}
	req.Params.Name = tool
	req.Params.Arguments = args
	res, err := c.CallTool(ctx, req)
	if err != nil {
		e.drop(c)
		return "", fmt.Errorf("%w: %s: %v", ErrUnavailable, tool, err)
	}
	text := resultText(res)
	if res.IsError {
		return "", fmt.Errorf("%w: %s: %s", ErrUnavailable, tool, preview(text, 240))
	}
	return text, nil
}

func resultText(res *mcp.CallToolResult) string {
	for _, c := range res.Content {
		if t, ok := c.(mcp.TextContent); ok {
			return t.Text
		}
	}
	if res.StructuredContent != nil {
		if b, err := json.Marshal(res.StructuredContent); err == nil {
			return string(b)
		}
	}
	return ""
}

type attribution struct {
	EntityIDs       []string        `json:"entity_ids"`
	EntityRevisions map[string]*int `json:"entity_revisions"`
	Error           string          `json:"error"`
}

func (e *mcpEvolve) Guidelines(ctx context.Context, namespace, task, sessionID string) ([]Guideline, error) {
	text, err := e.call(ctx, "get_guidelines_with_attribution", map[string]any{
		"task": task, "namespace_id": namespace, "session_id": sessionID,
	})
	if err != nil {
		return nil, err
	}
	var a attribution
	if err := json.Unmarshal([]byte(text), &a); err != nil {
		return nil, fmt.Errorf("%w: decode get_guidelines_with_attribution: %v", ErrUnavailable, err)
	}
	if a.Error != "" {
		return nil, fmt.Errorf("%w: %s", ErrUnavailable, a.Error)
	}
	if len(a.EntityIDs) == 0 {
		return nil, nil
	}
	// The attribution's prose is evolve's own prompt wording; the plugin owns the
	// block's format, so it takes the ids and reads the guidelines themselves.
	// list_entities rather than get_entity: get_entity refuses a guideline whose
	// agent_id differs from the caller's, and a caller cannot name one agent that
	// owns both unowned and agent-owned guidelines.
	byID, err := e.guidelineIndex(ctx, namespace)
	if err != nil {
		return nil, err
	}
	out := make([]Guideline, 0, len(a.EntityIDs))
	for _, id := range a.EntityIDs {
		g, ok := byID[id]
		if !ok {
			continue // removed between the two calls
		}
		g.Revision = a.EntityRevisions[id]
		out = append(out, g)
	}
	return out, nil
}

// guidelineIndex reads every guideline in namespace, keyed by id. The served
// namespace is a reviewed set, small by construction; maxServedGuidelines
// bounds the read regardless.
func (e *mcpEvolve) guidelineIndex(ctx context.Context, namespace string) (map[string]Guideline, error) {
	text, err := e.call(ctx, "list_entities", map[string]any{
		"namespace_id": namespace, "entity_types": []string{"guideline"},
		"include_content": true, "limit": maxServedGuidelines,
	})
	if err != nil {
		return nil, err
	}
	var page struct {
		Items []struct {
			ID       string         `json:"id"`
			Content  any            `json:"content"`
			Metadata map[string]any `json:"metadata"`
		} `json:"items"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(text), &page); err != nil {
		return nil, fmt.Errorf("%w: decode list_entities: %v", ErrUnavailable, err)
	}
	if page.Error != "" {
		return nil, fmt.Errorf("%w: list_entities: %s", ErrUnavailable, page.Error)
	}
	out := make(map[string]Guideline, len(page.Items))
	for _, item := range page.Items {
		g := Guideline{ID: item.ID}
		if s, ok := item.Content.(string); ok {
			g.Content = s
		} else {
			b, _ := json.Marshal(item.Content)
			g.Content = string(b)
		}
		g.Source, _ = item.Metadata["source_task_id"].(string)
		out[item.ID] = g
	}
	return out, nil
}

func (e *mcpEvolve) Learn(ctx context.Context, namespace string, t Trajectory) error {
	if err := e.ensure(ctx, namespace); err != nil {
		return err
	}
	raw, err := json.Marshal(t)
	if err != nil {
		return fmt.Errorf("marshal trajectory: %w", err)
	}
	var obj map[string]any
	_ = json.Unmarshal(raw, &obj)
	_, err = e.call(ctx, "process_trajectory", map[string]any{
		"trajectory": obj, "namespace_id": namespace, "processing_profile": e.profile,
	})
	return err
}

// ensure prepares the server for learning, once per session: the processing
// profile, and the learn namespace. process_trajectory requires the namespace to
// exist, and the MCP server has no tool that only creates one; every other tool
// that takes namespace_id creates it on first use, so a one-row listing does.
func (e *mcpEvolve) ensure(ctx context.Context, namespace string) error {
	e.mu.Lock()
	done := e.ensured
	e.mu.Unlock()
	if done {
		return nil
	}
	if _, err := e.call(ctx, "ensure_processing_profile", map[string]any{
		"profile_id": e.profile, "definition": profileDefinition,
	}); err != nil {
		return err
	}
	if _, err := e.call(ctx, "list_entities", map[string]any{"namespace_id": namespace, "limit": 1}); err != nil {
		return err
	}
	e.mu.Lock()
	e.ensured = true
	e.mu.Unlock()
	return nil
}

func (e *mcpEvolve) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.client != nil {
		_ = e.client.Close()
		e.client = nil
	}
}

func preview(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
