// Package evolve connects Cortex to altk-evolve, a service that learns
// guidelines from agent trajectories (https://github.com/AgentToolkit/altk-evolve).
//
// The plugin does two things, both optional:
//
//   - Inject. On a session's first main-agent inference request it asks evolve
//     for the active guidelines relevant to the session's task, caches them per
//     session, and appends them to the system instructions of every main-agent
//     request in that session as a delimited <evolve-guidelines> block. The
//     block is byte-stable across turns so prompt caching keeps working.
//   - Export. When a session has been idle for idle_after_seconds, the plugin
//     notifies evolve, which reads the session's new turns from Cortex's session
//     API and learns candidate guidelines from them.
//
// Evolve serves only guidelines a reviewer promoted to active; what it learns
// from sessions arrives as a candidate. That gate lives in evolve, not here:
// session content includes tool output, so nothing distilled from it should
// steer an agent unreviewed.
//
// Every failure fails open. Evolve unreachable, slow, or returning nonsense
// means the request goes upstream untouched; the guidelines are an
// improvement, never a dependency. Because parsers read the client's original
// bytes, the session archive never contains the injected block, so evolve does
// not learn from its own output.
package evolve

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"

	"github.com/rossoctl/cortex/core/pipeline"
	"github.com/rossoctl/cortex/core/plugins"
)

const name = "evolve"

type config struct {
	Endpoint   string   `json:"endpoint" required:"true" description:"Base URL of the evolve Cortex API (evolve cortex serve), e.g. http://127.0.0.1:8202."`
	Bearer     string   `json:"bearer" description:"Bearer token for the evolve API. Prefer bearer_file or ${ENV} expansion over a literal."`
	BearerFile string   `json:"bearer_file" description:"File holding the bearer token; ~ expands to the home directory."`
	Inject     *bool    `json:"inject" description:"Append active guidelines to main-agent requests." default:"true"`
	Export     *bool    `json:"export" description:"Notify evolve when a session goes idle so it can learn from it." default:"true"`
	Agents     []string `json:"agents" description:"Client names to act on (claude-code, codex, bob, opencode). Empty = all recognised agents."`
	TopK       int      `json:"top_k" description:"Most guidelines to inject per session." default:"8"`
	TimeoutMs  int      `json:"timeout_ms" description:"Per-call timeout for evolve. The first request of a session waits up to this long." default:"1500"`
	CacheTTL   int      `json:"cache_ttl_seconds" description:"How long a session's guidelines are reused before refetching (how soon a retired guideline stops)." default:"1800"`
	RetryAfter int      `json:"retry_after_seconds" description:"After a failed fetch, how long to inject nothing before trying again." default:"60"`
	IdleAfter  int      `json:"idle_after_seconds" description:"Quiet time after which a session is reported to evolve for learning." default:"900"`
	MaxCached  int      `json:"max_sessions" description:"Most sessions whose guidelines are cached." default:"1024"`
}

func (c *config) applyDefaults() {
	t := true
	if c.Inject == nil {
		c.Inject = &t
	}
	if c.Export == nil {
		c.Export = &t
	}
	if c.TopK == 0 {
		c.TopK = 8
	}
	if c.TimeoutMs == 0 {
		c.TimeoutMs = 1500
	}
	if c.CacheTTL == 0 {
		c.CacheTTL = 1800
	}
	if c.RetryAfter == 0 {
		c.RetryAfter = 60
	}
	if c.IdleAfter == 0 {
		c.IdleAfter = 900
	}
	if c.MaxCached == 0 {
		c.MaxCached = 1024
	}
}

func (c *config) validate() error {
	if !strings.HasPrefix(c.Endpoint, "http://") && !strings.HasPrefix(c.Endpoint, "https://") {
		return fmt.Errorf("endpoint must be an http(s) URL, got %q", c.Endpoint)
	}
	if c.Bearer != "" && c.BearerFile != "" {
		return fmt.Errorf("set bearer or bearer_file, not both")
	}
	if c.TopK < 1 || c.TopK > 50 {
		return fmt.Errorf("top_k must be between 1 and 50")
	}
	if c.TimeoutMs < 50 {
		return fmt.Errorf("timeout_ms must be at least 50")
	}
	if c.CacheTTL < 1 || c.RetryAfter < 1 || c.IdleAfter < 1 || c.MaxCached < 1 {
		return fmt.Errorf("cache_ttl_seconds, retry_after_seconds, idle_after_seconds and max_sessions must be positive")
	}
	return nil
}

// cached is one session's guidelines. block is pre-rendered so every request
// in the session appends identical bytes.
type cached struct {
	guidelines []Guideline
	block      string
	fetchedAt  time.Time
	expires    time.Time
	failed     bool
}

type seen struct {
	at    time.Time
	agent string
}

// Plugin is the evolve plugin.
type Plugin struct {
	cfg    config
	raw    json.RawMessage
	client Evolve
	agents map[string]struct{}
	now    func() time.Time

	mu       sync.Mutex
	cache    map[string]*cached
	order    []string // cache insertion order, for eviction
	inflight map[string]*sync.Mutex
	lastSeen map[string]*seen

	tick   time.Duration
	cancel context.CancelFunc
	done   chan struct{}

	m metrics
}

func New() *Plugin { return &Plugin{now: time.Now, tick: 30 * time.Second} }

func init() {
	plugins.RegisterPlugin(name, func() pipeline.Plugin { return New() })
}

func (p *Plugin) Name() string { return name }

func (p *Plugin) Capabilities() pipeline.PluginCapabilities {
	return pipeline.PluginCapabilities{
		ReadsBody:         true,
		WritesRequestBody: true,
		RequiresAny:       []string{"inference-parser"},
		Description:       "Injects reviewed altk-evolve guidelines; reports idle sessions for learning.",
	}
}

// ConfigSchema implements pipeline.SchemaProvider.
func (p *Plugin) ConfigSchema() []pipeline.FieldSchema { return pipeline.SchemaOf(config{}) }

// RawConfig implements pipeline.RawConfigProvider.
func (p *Plugin) RawConfig() json.RawMessage { return p.raw }

func (p *Plugin) Configure(raw json.RawMessage) error {
	var c config
	if len(raw) > 0 {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&c); err != nil {
			return fmt.Errorf("evolve config: %w", err)
		}
	}
	c.applyDefaults()
	if err := c.validate(); err != nil {
		return fmt.Errorf("evolve config: %w", err)
	}
	bearer := c.Bearer
	if c.BearerFile != "" {
		b, err := os.ReadFile(expandHome(c.BearerFile))
		if err != nil {
			return fmt.Errorf("evolve config: bearer_file: %w", err)
		}
		bearer = strings.TrimSpace(string(b))
	}
	p.cfg = c
	p.raw = raw
	p.agents = make(map[string]struct{}, len(c.Agents))
	for _, a := range c.Agents {
		p.agents[a] = struct{}{}
	}
	p.cache = map[string]*cached{}
	p.inflight = map[string]*sync.Mutex{}
	p.lastSeen = map[string]*seen{}
	if p.client == nil {
		p.client = newHTTPClient(strings.TrimRight(c.Endpoint, "/"), bearer, time.Duration(c.TimeoutMs)*time.Millisecond)
	}
	return nil
}

func expandHome(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(path, "~"))
		}
	}
	return path
}

// Init starts the idle-session notifier.
func (p *Plugin) Init(context.Context) error {
	if !*p.cfg.Export {
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	p.cancel = cancel
	p.done = make(chan struct{})
	go p.notifyLoop(ctx)
	return nil
}

// Shutdown stops the notifier. Sessions not yet reported are picked up later by
// `evolve sync cortex`, so nothing is flushed here.
func (p *Plugin) Shutdown(ctx context.Context) error {
	if p.cancel == nil {
		return nil
	}
	p.cancel()
	select {
	case <-p.done:
	case <-ctx.Done():
	}
	return nil
}

// eligible reports why a request is out of scope, or "" when it is a
// main-agent inference request of a known session and allowed agent.
func (p *Plugin) eligible(pctx *pipeline.Context) (reason, sessionID, agent string) {
	if pctx.Headers != nil && pctx.Headers.Get(sentinelHeader) != "" {
		return "own_call", "", ""
	}
	inf := pctx.Extensions.Inference
	if inf == nil {
		return "no_inference_extension", "", ""
	}
	if pctx.Subagent() != "" || inf.AgentRole == pipeline.AgentRoleSubagent {
		return "subagent", "", ""
	}
	if len(inf.Tools) == 0 {
		// Claude Code's tool-less side calls (titles, summaries) are not the task.
		return "no_tools", "", ""
	}
	if pctx.Session == nil {
		return "no_session", "", ""
	}
	sessionID = pctx.Session.ID
	if sessionID == "" || sessionID == "default" || strings.HasPrefix(sessionID, "pending:") {
		return "no_session", "", ""
	}
	client := pctx.ClientInfo()
	if client == nil || client.Name == "" {
		return "unknown_agent", "", ""
	}
	if len(p.agents) > 0 {
		if _, ok := p.agents[client.Name]; !ok {
			return "agent_not_selected", "", ""
		}
	}
	return "", sessionID, client.Name
}

// event is published as SessionEvent.Plugins["evolve"]: which guidelines
// shaped this request, so agentop can show it.
type event struct {
	Injected  bool      `json:"injected"`
	IDs       []string  `json:"ids,omitempty"`
	Revisions []*int    `json:"revisions,omitempty"`
	Sources   []string  `json:"sources,omitempty"`
	Cached    bool      `json:"cached"`
	Error     string    `json:"error,omitempty"`
	FetchedAt time.Time `json:"fetchedAt,omitzero"`
}

func (p *Plugin) OnRequest(ctx context.Context, pctx *pipeline.Context) (action pipeline.Action) {
	action = pipeline.Action{Type: pipeline.Continue}
	if !*p.cfg.Inject {
		return action
	}
	defer func() {
		if r := recover(); r != nil {
			p.m.inc(&p.m.recovered)
			slog.Warn("evolve: recovered, forwarding original body", "panic", r)
			action = pipeline.Action{Type: pipeline.Continue}
		}
	}()
	d := dialectOf(pathOnly(pctx.Path))
	if d == dialectNone {
		pctx.Skip("path_not_inference")
		return action
	}
	reason, sessionID, agent := p.eligible(pctx)
	if reason != "" {
		pctx.Skip(reason)
		return action
	}
	if len(pctx.Body) == 0 || !gjson.ValidBytes(pctx.Body) {
		pctx.Skip("invalid_body")
		return action
	}

	entry, hit := p.guidelinesFor(ctx, sessionID, agent, pctx.Extensions.Inference.Messages)
	ev := event{Cached: hit, FetchedAt: entry.fetchedAt}
	if len(entry.guidelines) == 0 {
		reason := "no_guidelines"
		if entry.failed {
			reason = "evolve_unavailable"
			ev.Error = reason
		}
		p.publish(pctx, ev)
		pctx.Skip(reason)
		return action
	}
	out, err := injectBlock(pctx.Body, d, entry.block)
	if err != nil || !gjson.ValidBytes(out) {
		p.m.inc(&p.m.injectErrors)
		ev.Error = "unsupported_request_shape"
		p.publish(pctx, ev)
		pctx.Skip("unsupported_request_shape")
		return action
	}
	applied := pctx.SetBody(out)
	ev.Injected = applied
	for _, g := range entry.guidelines {
		ev.IDs = append(ev.IDs, g.ID)
		ev.Revisions = append(ev.Revisions, g.MemoryRevision)
		ev.Sources = append(ev.Sources, g.SourceTaskID)
	}
	p.publish(pctx, ev)
	if applied {
		p.m.inc(&p.m.injected)
	} else {
		p.m.inc(&p.m.projected)
	}
	return action
}

// guidelinesFor returns the session's cached guidelines, fetching them when
// absent or expired. One fetch per session at a time: concurrent requests of
// the same session wait for it rather than each calling evolve.
func (p *Plugin) guidelinesFor(ctx context.Context, sessionID, agent string, messages []pipeline.InferenceMessage) (*cached, bool) {
	p.mu.Lock()
	if c, ok := p.cache[sessionID]; ok && p.now().Before(c.expires) {
		p.mu.Unlock()
		p.m.inc(&p.m.cacheHits)
		return c, true
	}
	lock, ok := p.inflight[sessionID]
	if !ok {
		lock = &sync.Mutex{}
		p.inflight[sessionID] = lock
	}
	p.mu.Unlock()

	lock.Lock()
	defer lock.Unlock()
	p.mu.Lock()
	if c, ok := p.cache[sessionID]; ok && p.now().Before(c.expires) {
		p.mu.Unlock()
		p.m.inc(&p.m.cacheHits)
		return c, true
	}
	previous := p.cache[sessionID]
	p.mu.Unlock()

	texts := make([]messageText, len(messages))
	for i, m := range messages {
		texts[i] = messageText{role: m.Role, content: m.Content}
	}
	entry := &cached{fetchedAt: p.now()}
	if task := taskText(texts); task == "" {
		entry.expires = p.now().Add(time.Duration(p.cfg.RetryAfter) * time.Second)
	} else {
		callCtx, cancel := context.WithTimeout(ctx, time.Duration(p.cfg.TimeoutMs)*time.Millisecond)
		gs, err := p.client.Guidelines(callCtx, GuidelinesRequest{Task: task, SessionID: sessionID, Agent: agent, TopK: p.cfg.TopK})
		cancel()
		p.m.inc(&p.m.fetches)
		switch {
		case err != nil:
			p.m.inc(&p.m.fetchErrors)
			slog.Warn("evolve: guideline fetch failed; forwarding requests unchanged", "session", sessionID, "err", err)
			entry.failed = true
			entry.expires = p.now().Add(time.Duration(p.cfg.RetryAfter) * time.Second)
		default:
			if len(gs) > p.cfg.TopK {
				gs = gs[:p.cfg.TopK]
			}
			entry.guidelines = gs
			entry.expires = p.now().Add(time.Duration(p.cfg.CacheTTL) * time.Second)
			if len(gs) > 0 {
				entry.block = formatBlock(gs)
			}
		}
	}
	// Keep serving what the session already had through a transient failure,
	// rather than dropping guidelines mid-session and breaking the prompt cache.
	if entry.failed && previous != nil && len(previous.guidelines) > 0 {
		entry.guidelines, entry.block, entry.failed = previous.guidelines, previous.block, false
	}
	p.store(sessionID, entry)
	return entry, false
}

func (p *Plugin) store(sessionID string, entry *cached) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.cache[sessionID]; !ok {
		p.order = append(p.order, sessionID)
	}
	p.cache[sessionID] = entry
	for len(p.order) > p.cfg.MaxCached {
		evict := p.order[0]
		p.order = p.order[1:]
		delete(p.cache, evict)
		delete(p.inflight, evict)
	}
}

func (p *Plugin) publish(pctx *pipeline.Context, ev event) {
	if pctx.Extensions.Custom == nil {
		pctx.Extensions.Custom = map[string]any{}
	}
	pctx.Extensions.Custom[name+pipeline.PluginEventSuffix] = ev
}

func (p *Plugin) OnResponse(context.Context, *pipeline.Context) pipeline.Action {
	return pipeline.Action{Type: pipeline.Continue}
}

// OnFinish records session activity for the idle notifier. It runs after the
// response is on the wire, so it only touches a map.
func (p *Plugin) OnFinish(_ context.Context, pctx *pipeline.Context) {
	if !*p.cfg.Export {
		return
	}
	if reason, sessionID, agent := p.eligible(pctx); reason == "" {
		p.mu.Lock()
		p.lastSeen[sessionID] = &seen{at: p.now(), agent: agent}
		p.mu.Unlock()
	}
}

func (p *Plugin) notifyLoop(ctx context.Context) {
	defer close(p.done)
	t := time.NewTicker(p.tick)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			p.notifyIdle(ctx)
		}
	}
}

// notifyIdle reports each session idle for idle_after_seconds once per idle
// period. A failed report is retried on the next tick; evolve's per-session
// cursor makes a duplicate report harmless.
func (p *Plugin) notifyIdle(ctx context.Context) {
	cutoff := p.now().Add(-time.Duration(p.cfg.IdleAfter) * time.Second)
	type due struct{ sessionID, agent string }
	var pending []due
	p.mu.Lock()
	for sid, s := range p.lastSeen {
		if !s.at.After(cutoff) {
			pending = append(pending, due{sid, s.agent})
		}
	}
	p.mu.Unlock()
	for _, d := range pending {
		callCtx, cancel := context.WithTimeout(ctx, time.Duration(p.cfg.TimeoutMs)*time.Millisecond)
		err := p.client.NotifyIdle(callCtx, d.sessionID, d.agent)
		cancel()
		if err != nil {
			p.m.inc(&p.m.notifyErrors)
			slog.Warn("evolve: idle-session notification failed; will retry", "session", d.sessionID, "err", err)
			continue
		}
		p.m.inc(&p.m.notified)
		p.mu.Lock()
		// Reported once per idle period: new activity records the session
		// again, and only then is it reported again.
		if s, ok := p.lastSeen[d.sessionID]; ok && !s.at.After(cutoff) {
			delete(p.lastSeen, d.sessionID)
		}
		p.mu.Unlock()
	}
}

// Metrics implements pipeline.MetricsProvider.
func (p *Plugin) Metrics() []pipeline.Metric { return p.m.snapshot() }
