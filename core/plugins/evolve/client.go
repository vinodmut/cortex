package evolve

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// Evolve is the plugin's view of the altk-evolve service. The concrete
// implementation (httpClient) calls `evolve cortex serve`; tests inject a fake.
type Evolve interface {
	// Guidelines returns the active guidelines relevant to a task.
	Guidelines(ctx context.Context, in GuidelinesRequest) ([]Guideline, error)
	// NotifyIdle tells evolve a session has gone quiet so it can learn from the
	// session's new turns. Evolve reads the turns from the session API itself.
	NotifyIdle(ctx context.Context, sessionID, agent string) error
}

// GuidelinesRequest is POST /api/cortex/guidelines.
type GuidelinesRequest struct {
	Task      string `json:"task"`
	SessionID string `json:"session_id,omitempty"`
	Agent     string `json:"agent,omitempty"`
	TopK      int    `json:"top_k,omitempty"`
}

// Guideline is one served guideline plus the provenance agentop shows for it.
type Guideline struct {
	ID             string `json:"id"`
	Content        string `json:"content"`
	Rationale      string `json:"rationale,omitempty"`
	MemoryRevision *int   `json:"memory_revision,omitempty"`
	SourceTaskID   string `json:"source_task_id,omitempty"`
	CreationMode   string `json:"creation_mode,omitempty"`
}

type guidelinesResponse struct {
	NamespaceID string      `json:"namespace_id"`
	Guidelines  []Guideline `json:"guidelines"`
}

// ErrUnavailable wraps every failure to get an answer from evolve: transport
// errors, timeouts, non-2xx statuses and undecodable bodies. The plugin fails
// open on all of them.
var ErrUnavailable = errors.New("evolve unavailable")

// sentinelHeader marks the plugin's own calls so that, should one ever loop
// back through the listener, OnRequest skips it instead of recursing.
const sentinelHeader = "X-Evolve-Cortex"

const maxResponseBytes = 1 << 20

// httpClient calls the evolve Cortex API with a standalone http.Client that
// never uses a proxy: Cortex is usually also the machine's HTTPS_PROXY, and the
// sentinel header is belt-and-suspenders on top of that.
type httpClient struct {
	base   string
	bearer string
	client *http.Client
}

func newHTTPClient(endpoint, bearer string, timeout time.Duration) *httpClient {
	return &httpClient{
		base:   endpoint,
		bearer: bearer,
		client: &http.Client{
			Timeout:   timeout,
			Transport: &http.Transport{Proxy: nil, MaxIdleConnsPerHost: 4, IdleConnTimeout: 90 * time.Second},
		},
	}
}

func (c *httpClient) Guidelines(ctx context.Context, in GuidelinesRequest) ([]Guideline, error) {
	var out guidelinesResponse
	if err := c.post(ctx, "/api/cortex/guidelines", in, &out); err != nil {
		return nil, err
	}
	return out.Guidelines, nil
}

func (c *httpClient) NotifyIdle(ctx context.Context, sessionID, agent string) error {
	path := "/api/cortex/sessions/" + url.PathEscape(sessionID) + "/ingest"
	return c.post(ctx, path, map[string]string{"agent": agent}, nil)
}

func (c *httpClient) post(ctx context.Context, path string, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return fmt.Errorf("marshal evolve request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(sentinelHeader, "1")
	if c.bearer != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearer)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("%w: status %d: %s", ErrUnavailable, resp.StatusCode, preview(string(respBody), 240))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("%w: decode response: %v", ErrUnavailable, err)
	}
	return nil
}

func preview(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
