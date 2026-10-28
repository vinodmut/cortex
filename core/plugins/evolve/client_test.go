package evolve

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHTTPClient_Guidelines(t *testing.T) {
	var gotAuth, gotSentinel string
	var gotBody GuidelinesRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/cortex/guidelines" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		gotAuth, gotSentinel = r.Header.Get("Authorization"), r.Header.Get(sentinelHeader)
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = io.WriteString(w, `{"namespace_id":"cortex","guidelines":[{"id":"g1","content":"c","memory_revision":3,"creation_mode":"auto-cortex"}]}`)
	}))
	defer srv.Close()

	gs, err := newHTTPClient(srv.URL, "tok", time.Second).Guidelines(context.Background(), GuidelinesRequest{Task: "t", TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer tok" || gotSentinel != "1" || gotBody.Task != "t" || gotBody.TopK != 2 {
		t.Fatalf("auth=%q sentinel=%q body=%+v", gotAuth, gotSentinel, gotBody)
	}
	if len(gs) != 1 || gs[0].ID != "g1" || *gs[0].MemoryRevision != 3 || gs[0].CreationMode != "auto-cortex" {
		t.Fatalf("guidelines = %+v", gs)
	}
}

func TestHTTPClient_NotifyIdleEscapesSessionID(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.EscapedPath()
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()
	if err := newHTTPClient(srv.URL, "", time.Second).NotifyIdle(context.Background(), "a/b c", "claude-code"); err != nil {
		t.Fatal(err)
	}
	if gotPath != "/api/cortex/sessions/a%2Fb%20c/ingest" {
		t.Fatalf("path = %q", gotPath)
	}
}

func TestHTTPClient_ErrorsAreUnavailable(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusServiceUnavailable, http.StatusInternalServerError} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, strings.Repeat("x", 500), status)
		}))
		_, err := newHTTPClient(srv.URL, "", time.Second).Guidelines(context.Background(), GuidelinesRequest{Task: "t"})
		srv.Close()
		if !errors.Is(err, ErrUnavailable) || len(err.Error()) > 320 {
			t.Errorf("status %d: err = %v", status, err)
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, "not json") }))
	defer srv.Close()
	if _, err := newHTTPClient(srv.URL, "", time.Second).Guidelines(context.Background(), GuidelinesRequest{Task: "t"}); !errors.Is(err, ErrUnavailable) {
		t.Errorf("decode error not wrapped: %v", err)
	}
}

func TestHTTPClient_Timeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()
	_, err := newHTTPClient(srv.URL, "", 20*time.Millisecond).Guidelines(context.Background(), GuidelinesRequest{Task: "t"})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v", err)
	}
}

func TestHTTPClient_IgnoresProxyEnvironment(t *testing.T) {
	t.Setenv("HTTP_PROXY", "http://127.0.0.1:1")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"guidelines":[]}`)
	}))
	defer srv.Close()
	if _, err := newHTTPClient(srv.URL, "", time.Second).Guidelines(context.Background(), GuidelinesRequest{Task: "t"}); err != nil {
		t.Fatalf("request went through the proxy: %v", err)
	}
}
