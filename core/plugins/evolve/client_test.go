package evolve

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// fakeEvolveServer serves the evolve MCP tools the plugin calls, with canned
// answers shaped like the stock server's, over a real SSE transport.
type fakeEvolveServer struct {
	mu        sync.Mutex
	calls     []mcp.CallToolRequest
	headers   []http.Header
	failLearn bool
	listErr   bool
}

func (f *fakeEvolveServer) record(req mcp.CallToolRequest) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, req)
	f.headers = append(f.headers, req.Header)
}

func (f *fakeEvolveServer) start(t *testing.T) string {
	t.Helper()
	s := server.NewMCPServer("entities", "test")
	text := func(v any) *mcp.CallToolResult {
		b, _ := json.Marshal(v)
		return mcp.NewToolResultText(string(b))
	}
	s.AddTool(mcp.NewTool("get_guidelines_with_attribution"), func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		f.record(req)
		return text(map[string]any{"text": "prose", "entity_ids": []string{"7", "9"}, "entity_revisions": map[string]any{"7": 2, "9": nil}, "namespace_id": "cortex"}), nil
	})
	s.AddTool(mcp.NewTool("ensure_processing_profile"), func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		f.record(req)
		return text(map[string]any{"id": "cortex", "revision": 1}), nil
	})
	s.AddTool(mcp.NewTool("list_entities"), func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		f.record(req)
		if f.listErr {
			return text(map[string]any{"error": "invalid cursor"}), nil
		}
		// 7 is unowned, 9 was learned from a Claude Code session; 8 is not attributed.
		return text(map[string]any{"items": []any{
			map[string]any{"id": "7", "content": "guideline 7", "metadata": map[string]any{"source_task_id": "cortex:s:7"}},
			map[string]any{"id": "8", "content": "guideline 8", "metadata": map[string]any{}},
			map[string]any{"id": "9", "content": "guideline 9", "metadata": map[string]any{"agent_id": "claude-code"}},
		}}), nil
	})
	s.AddTool(mcp.NewTool("process_trajectory"), func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		f.record(req)
		if f.failLearn {
			return mcp.NewToolResultError("Namespace not found"), nil
		}
		return text(map[string]any{"operation_id": "op", "updates": []any{}}), nil
	})
	ts := server.NewTestServer(s)
	t.Cleanup(ts.Close)
	return ts.URL + "/sse"
}

func (f *fakeEvolveServer) tools() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var names []string
	for _, c := range f.calls {
		names = append(names, c.Params.Name)
	}
	return names
}

func TestMCPEvolve_GuidelinesReadsEachAttributedEntity(t *testing.T) {
	f := &fakeEvolveServer{}
	e := newMCPEvolve(f.start(t), "cortex", time.Second)
	defer e.Close()

	gs, err := e.Guidelines(context.Background(), "cortex", "fix the test", "s1")
	if err != nil {
		t.Fatal(err)
	}
	if len(gs) != 2 || gs[0].ID != "7" || gs[0].Content != "guideline 7" || *gs[0].Revision != 2 || gs[0].Source != "cortex:s:7" ||
		gs[1].ID != "9" || gs[1].Content != "guideline 9" || gs[1].Revision != nil {
		t.Fatalf("guidelines = %+v", gs)
	}
	args := f.calls[0].GetArguments()
	if args["task"] != "fix the test" || args["namespace_id"] != "cortex" || args["session_id"] != "s1" {
		t.Fatalf("args = %v", args)
	}
	if a := f.calls[1].GetArguments(); f.calls[1].Params.Name != "list_entities" || a["namespace_id"] != "cortex" || a["include_content"] != true {
		t.Fatalf("second call = %s %v, want list_entities of the serve namespace with content", f.calls[1].Params.Name, a)
	}
}

func TestMCPEvolve_LearnPreparesServerOnce(t *testing.T) {
	f := &fakeEvolveServer{}
	e := newMCPEvolve(f.start(t), "cortex", time.Second)
	defer e.Close()
	tr := Trajectory{
		Messages: []map[string]any{{"role": "user", "content": "x"}},
		Batch:    &Batch{Source: "cortex", ConversationID: "s1", BatchID: "1-1", Scope: "cortex"},
	}
	for range 2 {
		if err := e.Learn(context.Background(), "cortex-candidates", tr); err != nil {
			t.Fatal(err)
		}
	}
	got := f.tools()
	if len(got) != 4 || got[0] != "ensure_processing_profile" || got[1] != "list_entities" || got[2] != "process_trajectory" || got[3] != "process_trajectory" {
		t.Fatalf("tools called = %v", got)
	}
	if f.calls[1].GetArguments()["namespace_id"] != "cortex-candidates" {
		t.Fatal("learn namespace not created before processing")
	}
	args := f.calls[2].GetArguments()
	if args["namespace_id"] != "cortex-candidates" || args["processing_profile"] != "cortex" {
		t.Fatalf("args = %v", args)
	}
	batch := args["trajectory"].(map[string]any)["batch"].(map[string]any)
	if batch["conversation_id"] != "s1" || batch["batch_id"] != "1-1" {
		t.Fatalf("batch = %v", batch)
	}
}

func TestMCPEvolve_ToolErrorsAreUnavailable(t *testing.T) {
	f := &fakeEvolveServer{failLearn: true}
	e := newMCPEvolve(f.start(t), "cortex", time.Second)
	defer e.Close()
	err := e.Learn(context.Background(), "cortex-candidates", Trajectory{Messages: []map[string]any{}})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v", err)
	}
	f.listErr = true
	if _, err := e.Guidelines(context.Background(), "cortex", "t", "s"); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("list error = %v", err)
	}
}

func TestMCPEvolve_UnreachableIsUnavailable(t *testing.T) {
	e := newMCPEvolve("http://127.0.0.1:1/sse", "cortex", time.Second)
	defer e.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := e.Guidelines(ctx, "cortex", "t", "s"); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v", err)
	}
}

func TestMCPEvolve_IgnoresProxyEnvironment(t *testing.T) {
	t.Setenv("HTTP_PROXY", "http://127.0.0.1:1")
	f := &fakeEvolveServer{}
	e := newMCPEvolve(f.start(t), "cortex", time.Second)
	defer e.Close()
	if _, err := e.Guidelines(context.Background(), "cortex", "t", "s"); err != nil {
		t.Fatalf("request went through the proxy: %v", err)
	}
}
