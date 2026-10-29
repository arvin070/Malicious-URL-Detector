/* URLGuard web client.
 *
 * Two rules govern this file:
 *
 * 1. The UI renders exactly what the API returned, and says "Not available"
 *    when it returned nothing. No defaults, no placeholders, no inferred
 *    values. If a number is on screen, the server measured it.
 *
 * 2. No secret is stored here. The access token is short-lived and held in
 *    memory; only the refresh token is persisted, and only when the user asked
 *    to stay signed in. There is no API key in this bundle, because shipping
 *    one to a browser publishes it.
 */

const $ = (id) => document.getElementById(id);
const api = {
  access: null,
  refresh: null,
  user: null,
};

const STORE_KEY = 'urlguard.session';

/* ------------------------------------------------------------- utilities */

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

function fmtNum(v) {
  if (typeof v !== 'number') return escapeHtml(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}

function fmtTime(sec) {
  if (!sec) return 'Not available';
  return new Date(sec * 1000).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function relTime(sec) {
  if (!sec) return '';
  const s = Math.round(Date.now() / 1000 - sec);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function fieldValue(f) {
  if (!f || typeof f !== 'object') return 'Not available';
  if (f.available && f.value !== null && f.value !== '') return escapeHtml(f.value);
  return '<span class="unavailable">Not available</span>';
}

function notEnough(msg) {
  return `<div class="insufficient"><strong>Not enough data yet.</strong>
    <p>${escapeHtml(msg)}</p></div>`;
}

function setBusy(btn, on, label) {
  if (!btn) return;
  btn.disabled = on;
  btn.classList.toggle('is-loading', on);
  const l = btn.querySelector('.btn-label');
  if (l && label) l.textContent = on ? label[1] : label[0];
}

/* ------------------------------------------------------------- transport */

async function call(path, { method = 'GET', body, auth = true, retry = true } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (auth && api.access) headers.Authorization = `Bearer ${api.access}`;
  const res = await fetch(path, {
    method, headers, body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch (e) { /* empty body */ }

  // One transparent refresh attempt, so a 30-minute access token does not
  // interrupt someone mid-task.
  if (res.status === 401 && retry && api.refresh &&
      ['token_expired', 'token_invalid', 'auth_required'].includes(data.code)) {
    const ok = await tryRefresh();
    if (ok) return call(path, { method, body, auth, retry: false });
  }
  if (!res.ok) {
    const err = new Error(data.error || `Request failed (${res.status})`);
    err.status = res.status;
    err.code = data.code;
    err.data = data;
    throw err;
  }
  return data;
}

async function tryRefresh() {
  if (!api.refresh) return false;
  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: api.refresh }),
    });
    if (!res.ok) throw new Error('refresh failed');
    const d = await res.json();
    adoptSession(d, !!loadStored());
    return true;
  } catch (e) {
    clearSession();
    return false;
  }
}

/* --------------------------------------------------------------- session */

function loadStored() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

function adoptSession(d, persist) {
  api.access = d.access_token;
  api.refresh = d.refresh_token;
  api.user = d.user || api.user;
  if (persist) {
    try {
      // Only the refresh token is persisted, and only on request. The access
      // token stays in memory so a closed tab drops it.
      localStorage.setItem(STORE_KEY, JSON.stringify({ refresh: api.refresh }));
    } catch (e) { /* private mode */ }
  }
}

function clearSession() {
  api.access = api.refresh = api.user = null;
  try { localStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------ auth screen */

const AUTH_VIEWS = {
  login: 'login-form', register: 'register-form',
  forgot: 'forgot-form', reset: 'reset-form',
};

function showAuthView(name) {
  Object.entries(AUTH_VIEWS).forEach(([k, id]) => { $(id).hidden = k !== name; });
  $('auth-error').hidden = true;
  $('auth-notice').hidden = true;
}

function authError(msg) {
  $('auth-error').textContent = msg;
  $('auth-error').hidden = false;
  $('auth-notice').hidden = true;
}

function authNotice(msg) {
  $('auth-notice').textContent = msg;
  $('auth-notice').hidden = false;
  $('auth-error').hidden = true;
}

document.querySelectorAll('[data-auth-view]').forEach((b) =>
  b.addEventListener('click', () => showAuthView(b.dataset.authView)));

document.querySelectorAll('.pw-toggle').forEach((b) =>
  b.addEventListener('click', () => {
    const inp = $(b.dataset.for);
    const show = inp.type === 'password';
    inp.type = show ? 'text' : 'password';
    b.textContent = show ? 'Hide' : 'Show';
  }));

$('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  setBusy(btn, true, ['Sign in', 'Signing in']);
  try {
    const d = await call('/api/auth/login', {
      method: 'POST', auth: false,
      body: { email: $('login-email').value, password: $('login-password').value },
    });
    adoptSession(d, $('login-remember').checked);
    await enterApp();
  } catch (err) {
    authError(err.message);
  } finally { setBusy(btn, false, ['Sign in', 'Signing in']); }
});

$('register-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  if ($('reg-password').value !== $('reg-confirm').value) {
    authError('Passwords do not match.'); return;
  }
  setBusy(btn, true, ['Create account', 'Creating']);
  try {
    const d = await call('/api/auth/register', {
      method: 'POST', auth: false,
      body: {
        name: $('reg-name').value, email: $('reg-email').value,
        password: $('reg-password').value,
        confirm_password: $('reg-confirm').value,
      },
    });
    adoptSession(d, true);
    await enterApp();
  } catch (err) {
    authError(err.message);
  } finally { setBusy(btn, false, ['Create account', 'Creating']); }
});

$('forgot-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  setBusy(btn, true, ['Request reset', 'Sending']);
  try {
    const d = await call('/api/auth/forgot', {
      method: 'POST', auth: false, body: { email: $('forgot-email').value },
    });
    authNotice(`${d.message} ${d.delivery}`);
  } catch (err) {
    authError(err.message);
  } finally { setBusy(btn, false, ['Request reset', 'Sending']); }
});

$('reset-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  setBusy(btn, true, ['Set new password', 'Saving']);
  try {
    await call('/api/auth/reset', {
      method: 'POST', auth: false,
      body: { token: $('reset-token').value, new_password: $('reset-password').value },
    });
    showAuthView('login');
    authNotice('Password updated. Please sign in.');
  } catch (err) {
    authError(err.message);
  } finally { setBusy(btn, false, ['Set new password', 'Saving']); }
});

$('skip-auth').addEventListener('click', async () => {
  clearSession();
  await enterApp(true);
});

/* ------------------------------------------------------------------ views */

function gotoView(name) {
  document.querySelectorAll('.tab').forEach((t) => {
    const on = t.dataset.view === name;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  document.querySelectorAll('.view').forEach((v) => {
    v.classList.toggle('is-active', v.id === `view-${name}`);
  });
  const loaders = {
    dashboard: loadDashboard, history: loadHistory, threats: loadThreats,
    analytics: loadAnalytics, profile: loadProfile, settings: loadSettings,
  };
  if (loaders[name]) loaders[name]();
}

document.querySelectorAll('.tab').forEach((t) =>
  t.addEventListener('click', () => gotoView(t.dataset.view)));
document.querySelectorAll('[data-goto]').forEach((b) =>
  b.addEventListener('click', () => gotoView(b.dataset.goto)));

async function enterApp(anonymous = false) {
  $('auth-screen').hidden = true;
  $('app').hidden = false;
  if (!anonymous && api.access) {
    try {
      const me = await call('/api/auth/me');
      api.user = me.user;
    } catch (e) { /* fall through as anonymous */ }
  }
  $('who').innerHTML = api.user
    ? `${escapeHtml(api.user.name)} <button class="linkish" id="quick-signout">Sign out</button>`
    : `<button class="linkish" id="quick-signin">Sign in</button>`;
  const so = $('quick-signout');
  if (so) so.addEventListener('click', doLogout);
  const si = $('quick-signin');
  if (si) si.addEventListener('click', () => {
    $('app').hidden = true; $('auth-screen').hidden = false; showAuthView('login');
  });

  try {
    const meta = await call('/api/meta', { auth: false });
    window.__meta = meta;
    $('footer-meta').textContent =
      `${meta.n_features} features · ${meta.model_version || 'model'} · ` +
      `threshold ${Math.round(meta.threshold * 100)}/100`;
    const badge = $('model-badge');
    if (badge) badge.textContent =
      `${meta.models.length} models · ${meta.model_version || ''}`;
  } catch (e) { /* header already rendered server-side */ }

  gotoView('dashboard');
}

async function doLogout() {
  try {
    await call('/api/auth/logout', {
      method: 'POST', body: { refresh_token: api.refresh },
    });
  } catch (e) { /* logging out locally regardless */ }
  clearSession();
  $('app').hidden = true;
  $('auth-screen').hidden = false;
  showAuthView('login');
  authNotice('You have been signed out.');
}

/* --------------------------------------------------------------- scanner */

const scanForm = $('scan-form');
scanForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const url = $('url-input').value.trim();
  if (!url) return;
  setBusy($('scan-btn'), true, ['Analyse', 'Analysing']);
  $('scan-error').hidden = true;
  try {
    const scope = (document.querySelector('input[name=scope]:checked') || {}).value || 'url';
    const analyzeWebsite = $('analyze-website').checked;
    const data = await call('/api/scan', {
      method: 'POST', body: { url, scope, analyze_website: analyzeWebsite },
    });
    renderResult(data);
  } catch (err) {
    $('scan-error').textContent = err.message;
    $('scan-error').hidden = false;
    $('result').hidden = true;
  } finally {
    setBusy($('scan-btn'), false, ['Analyse', 'Analysing']);
  }
});

$('clear-btn').addEventListener('click', () => {
  $('url-input').value = '';
  $('result').hidden = true;
  $('scan-error').hidden = true;
  $('url-input').focus();
});

function renderResult(data) {
  const a = data.assessment;
  $('result').hidden = false;
  $('assessment-card').className = `card assessment band-${a.risk_band.key}`;
  $('band-label').textContent = a.risk_band.label;
  $('band-desc').textContent = a.risk_band.description;
  $('risk-score').textContent = a.risk_score;
  $('confidence').textContent = `${a.confidence}%`;
  $('confidence-label').textContent = a.confidence_label ? `· ${a.confidence_label}` : '';
  $('confidence-label').title = a.confidence_basis || '';
  $('scale-marker').style.left = `${a.risk_score}%`;
  $('result-url').textContent = data.subject || data.url;
  $('disclaimer').textContent = a.disclaimer;

  const warn = $('scope-warning');
  if (a.scope_warning) { warn.textContent = a.scope_warning; warn.hidden = false; }
  else warn.hidden = true;

  // --- detection breakdown ------------------------------------------------
  const rows = a.breakdown || [];
  $('breakdown').innerHTML = rows.map((r) => `
    <div class="bd-row ${r.available ? '' : 'bd-missing'}"${
      r.reason ? ` title="${escapeHtml(r.reason)}"` : ''}>
      <span class="bd-label">${escapeHtml(r.label)}</span>
      <span class="bd-value bd-${String(r.value).toLowerCase().replace(/[^a-z]/g, '')}">
        ${r.available ? escapeHtml(r.value)
          : `<span class="unavailable">${escapeHtml(r.value)}</span>`}
        ${r.score != null ? `<i class="bd-score">${r.score}</i>` : ''}
      </span>
    </div>`).join('');
  const fz = a.fusion || {};
  $('fusion-note').textContent = fz.method
    ? `${fz.method}.` + (a.url_only_risk_score != null && a.url_only_risk_score !== a.risk_score
        ? ` URL alone scored ${a.url_only_risk_score}/100.` : '')
    : '';
  renderFusionModel(a, fz);
  renderPageLayers(a);

  $('meta-row').innerHTML = [
    ['Scanned', fmtTime(data.scanned_at)],
    ['Detection time', data.detection_ms != null ? `${data.detection_ms} ms`
      : '<span class="unavailable">Not available</span>'],
    ['Model', escapeHtml(data.model_version || data.best_model_name)],
    ['Website model', data.web_model_version
      ? escapeHtml(data.web_model_version)
      : '<span class="unavailable">Not used</span>'],
    ['Dataset', escapeHtml(data.dataset_version || 'Not available')],
    ['Scope', data.scope === 'hostname' ? 'Hostname only' : 'Full URL'],
    ['Threshold', `${a.threshold_score}/100`],
  ].map(([k, v]) => `<div class="meta-item"><span>${k}</span><strong>${v}</strong></div>`).join('');

  const basis = a.signal_basis || {};
  $('signal-basis').innerHTML = basis.available
    ? `Thresholds are the p${basis.percentile} value of ${Number(basis.n_benign).toLocaleString()} known-benign URLs. A measurement is listed as a risk signal only when exceeding it is at least ${basis.min_lift}× more common among phishing samples.`
    : '<span class="unavailable">Signal thresholds unavailable — run <code>python src/build_signal_thresholds.py</code>.</span>';

  renderSignals('signals-warning', a.signals.warning, 'warn', '⚠');
  renderSignals('signals-positive', a.signals.positive, 'ok', '✓');
  renderSignals('signals-neutral', a.signals.neutral, 'neutral', '·');
  $('neutral-wrap').hidden = a.signals.neutral.length === 0;
  $('neutral-summary').textContent =
    `Observations not treated as risk signals (${a.signals.neutral.length})`;
  if (!a.signals.warning.length && !a.signals.positive.length) {
    $('signals-warning').innerHTML =
      '<p class="muted">No security signals were raised for this URL.</p>';
  }

  $('explanation-text').textContent = a.explanation_text;
  const ex = a.explanation;
  if (ex && ex.available && ex.contributions.length) {
    const max = Math.max(...ex.contributions.map((c) => Math.abs(c.contribution)));
    $('explanation-detail').innerHTML = `<div class="shap-list">${ex.contributions.map((c) => {
      const w = max ? (Math.abs(c.contribution) / max) * 100 : 0;
      const up = c.direction === 'increases';
      return `<div class="shap-row">
        <span class="shap-name">${escapeHtml(c.feature)}</span>
        <span class="shap-val">${fmtNum(c.value)}</span>
        <span class="shap-bar ${up ? 'up' : 'down'}"><i style="width:${w}%"></i></span>
        <span class="shap-contrib ${up ? 'up' : 'down'}">${c.contribution > 0 ? '+' : ''}${c.contribution}</span>
      </div>`;
    }).join('')}</div><p class="card-note">${escapeHtml(ex.method)}.</p>`;
  } else {
    $('explanation-detail').innerHTML =
      `<p class="muted">${escapeHtml((ex && ex.reason) || 'Detailed model explanation is not available for this scan.')}</p>`;
  }

  $('characteristics').innerHTML = a.characteristics.map((c) => `
    <div class="char-item"><span class="char-label">${escapeHtml(c.label)}</span>
    <span class="char-value">${escapeHtml(c.value)}${c.unit ? ` <span class="char-unit">${escapeHtml(c.unit)}</span>` : ''}</span></div>`).join('');

  const d = a.domain;
  $('domain-scope').textContent = d.scope_note ||
    'Hostname and domain are parsed from the URL text. The remaining fields would require contacting the network, which this system does not do.';
  $('domain-info').innerHTML = [
    ['Hostname', d.hostname], ['Registered domain', d.registered_domain],
    ['Public suffix (TLD)', d.tld], ['DNS resolution', d.dns_resolution],
    ['IP address', d.ip_address], ['Domain age', d.domain_age],
    ['Reputation', d.reputation], ['DNS classification', d.dns_classification],
  ].map(([k, f]) => `<div class="kv-row"${f && !f.available && f.reason ? ` title="${escapeHtml(f.reason)}"` : ''}>
      <span class="kv-key">${k}</span><span class="kv-val">${fieldValue(f)}</span></div>`).join('');

  const ag = data.model_agreement || {};
  $('agreement-note').textContent =
    `${ag.flagged} of ${ag.total} models placed this URL above the decision threshold of ${a.threshold_score}/100. The verdict follows the selected model (${data.best_model_name}).`;
  $('model-rows').innerHTML = data.models.map((m) => {
    const pct = Math.round(m.score * 100);
    return `<div class="model-row">
      <span class="model-name ${m.is_best ? 'is-best' : ''}">${escapeHtml(m.name)}${m.is_best ? ' <em>selected</em>' : ''}</span>
      <span class="model-meter"><i style="width:${pct}%" class="${m.label === 'malicious' ? 'hot' : ''}"></i>
      <b style="left:${a.threshold_score}%" title="decision threshold"></b></span>
      <span class="model-score">${pct}</span></div>`;
  }).join('');

  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* The calibrated meta-model's own account of itself. Shown only when it
   actually produced the headline, so the panel is never an explanation of a
   number that came from somewhere else. */
function renderFusionModel(a, fz) {
  const el = $('fusion-model');
  const m = fz.model;
  if (!m) { el.hidden = true; el.innerHTML = ''; return; }
  el.hidden = false;
  const cal = m.calibration || {};
  const coverage = `${m.inputs_present} of ${m.inputs_total}`;
  const missing = (m.inputs_missing || []);
  el.innerHTML = `
    <div class="fm-head">
      <span class="fm-tag">Fusion model</span>
      <span class="fm-name">${escapeHtml(m.model_name || '')}</span>
      <span class="fm-ver">${escapeHtml(m.model_version || '')}</span>
      ${m.undecided ? '<span class="fm-undecided">below decision margin</span>' : ''}
    </div>
    <p class="fm-basis">${escapeHtml(m.basis)}</p>
    <div class="fm-grid">
      <div><span>Calibrated probability</span><strong>${m.probability.toFixed(3)}</strong></div>
      <div><span>Decision threshold</span><strong>${m.threshold.toFixed(2)}</strong></div>
      <div><span>Evidence inputs used</span><strong>${coverage}</strong></div>
      <div><span>Calibration error (ECE)</span><strong>${
        cal.ece != null ? Number(cal.ece).toFixed(4)
          : '<span class="unavailable">Not available</span>'}</strong></div>
    </div>
    ${missing.length ? `<details class="fm-missing">
      <summary>${missing.length} input${missing.length === 1 ? '' : 's'} not measured for this page</summary>
      <p class="card-note">These were absent from the page or could not be read.
        They were left as missing rather than recorded as zero, so the model
        treats them as unmeasured instead of as a negative finding.</p>
      <div class="fm-chips">${missing.map((c) =>
        `<code>${escapeHtml(c)}</code>`).join('')}</div>
    </details>` : ''}
    <p class="card-note">${escapeHtml(m.limitation)}</p>`;
}

/* Per-layer findings from the retrieved page. Every layer states its own
   method and limitation; a layer that produced nothing says so rather than
   being omitted, because an absent section reads as an absent risk. */
function renderPageLayers(a) {
  const card = $('page-layers-card');
  const w = (a.layers || {}).website || {};
  if (!w.available) { card.hidden = true; $('page-layers').innerHTML = ''; return; }
  card.hidden = false;

  const blocks = [];

  // ---- HTML / page model
  blocks.push(layerBlock('Page structure', w.score, w.level, w.basis, [
    ['Page-pattern score', `${w.score}/100`],
    ['Layer threshold', w.threshold_score != null ? `${w.threshold_score}/100` : null],
    ['Model', w.model_version],
  ], null));

  // ---- JavaScript
  const js = w.javascript;
  if (js) {
    const hits = Object.entries(js.hits || {});
    blocks.push(layerBlock('JavaScript (static)', null, null, js.method, [
      ['Inline scripts analysed', js.features.js_inline_scripts],
      ['External scripts counted, not fetched', js.external_not_fetched],
      ['Distinct patterns matched', js.features.js_distinct_patterns],
      ['Bytes examined', js.analysed_bytes],
    ], hits.length
      ? `<div class="fm-chips">${hits.map(([k, v]) =>
          `<code>${escapeHtml(k)} ×${v}</code>`).join('')}</div>
         <p class="card-note">${escapeHtml(js.limitation)}</p>`
      : '<p class="muted">No monitored JavaScript pattern was matched.</p>'));
  } else {
    blocks.push(unavailableBlock('JavaScript (static)',
      'The script layer did not run for this page.'));
  }

  // ---- redirects
  const ra = w.redirect_analysis;
  if (ra) {
    const f = ra.features || {};
    const chain = (w.fetch && w.fetch.redirects) || [];
    blocks.push(layerBlock('Redirect chain', null, null, ra.method || null, [
      ['Hops', f.redirect_hops],
      ['Distinct domains', f.redirect_distinct_domains],
      ['Domain changed', yesNo(f.redirect_domain_changed)],
      ['HTTPS downgrade', yesNo(f.redirect_https_downgrades)],
      ['Started at a shortener', yesNo(f.redirect_start_is_shortener)],
    ], chain.length
      ? `<ol class="rd-chain">${chain.map((h) =>
          `<li><span class="rd-status">${escapeHtml(h.status)}</span>
             <code>${escapeHtml(h.to || '')}</code></li>`).join('')}</ol>`
      : '<p class="muted">The URL resolved without redirecting.</p>'));
  } else {
    blocks.push(unavailableBlock('Redirect chain',
      'The redirect layer did not run for this page.'));
  }

  // ---- structural fingerprint
  const fp = w.fingerprint;
  if (fp) {
    blocks.push(layerBlock('Structural fingerprint', null, null,
      'A hash of the page\u2019s tag sequence and form skeleton, used to find pages built from the same kit. It identifies structural similarity only and carries no verdict.', [
        // The hex form, not the integer: JSON numbers are doubles in the
        // browser and a 64-bit hash would be displayed rounded.
        ['Tag fingerprint', fp.tag_simhash_hex || ''],
        ['Form skeleton', fp.form_skeleton || '(no form)'],
        ['Nodes sampled', fp.node_sample],
      ], null));
  }

  $('page-layers').innerHTML = blocks.join('');
}

function yesNo(v) {
  if (v == null) return null;
  return Number(v) > 0 ? 'Yes' : 'No';
}

function layerBlock(title, score, level, note, rows, extra) {
  const kv = rows.filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `<div class="kv-row"><span class="kv-key">${escapeHtml(k)}</span>
      <span class="kv-val">${escapeHtml(v)}</span></div>`).join('');
  return `<div class="layer-block">
    <div class="layer-head"><h4>${escapeHtml(title)}</h4>
      ${score != null ? `<span class="layer-score">${score}/100</span>` : ''}
      ${level ? `<span class="layer-level">${escapeHtml(level)}</span>` : ''}</div>
    ${kv ? `<div class="kv">${kv}</div>` : ''}
    ${extra || ''}
    ${note ? `<p class="card-note">${escapeHtml(note)}</p>` : ''}
  </div>`;
}

function unavailableBlock(title, reason) {
  return `<div class="layer-block layer-missing">
    <div class="layer-head"><h4>${escapeHtml(title)}</h4>
      <span class="layer-level unavailable">Not available</span></div>
    <p class="card-note">${escapeHtml(reason)}</p></div>`;
}

function renderSignals(id, items, cls, glyph) {
  $(id).innerHTML = (items || []).map((s) => `
    <div class="signal signal-${cls}"><span class="signal-glyph" aria-hidden="true">${glyph}</span>
      <div><div class="signal-text">${escapeHtml(s.text)}</div>
      <div class="signal-evidence">${escapeHtml(s.evidence)}</div></div></div>`).join('');
}

/* ------------------------------------------------------------- dashboard */

function statTiles(s) {
  const blocked = s.high;
  return [
    ['URLs scanned', s.total, ''],
    ['Low risk', s.low, 'low'],
    ['Suspicious', s.suspicious, 'suspicious'],
    ['High risk', s.high, 'high'],
    ['Insufficient evidence', s.insufficient, 'insufficient'],
    ['Would be blocked', blocked, 'high'],
    ['Avg scan time', s.avg_detection_ms != null ? `${s.avg_detection_ms} ms` : null, ''],
  ].map(([label, value, key]) => `
    <div class="stat ${key ? `stat-${key}` : ''}">
      <div class="stat-value">${value == null ? '<span class="unavailable">Not available</span>' : escapeHtml(value)}</div>
      <div class="stat-label">${label}</div></div>`).join('');
}

async function loadDashboard() {
  let s;
  try { s = await call('/api/stats'); } catch (e) { s = null; }
  if (!s || !s.sufficient_data) {
    $('dash-stats').innerHTML = notEnough(
      'No scans have been recorded for this account yet. Run a scan and the counters will appear.');
    $('activity-chart').innerHTML = notEnough('Threat activity needs at least one recorded scan.');
    $('dash-recent').innerHTML = '<p class="muted">Not enough data yet.</p>';
    $('activity-note').textContent = '';
    return;
  }
  $('dash-stats').innerHTML = statTiles(s);
  $('activity-note').textContent =
    'Built from stored scans only. Each bar is one day of recorded activity.';
  $('activity-chart').innerHTML = seriesChart(s.series);

  const h = await call('/api/history?limit=8');
  $('dash-recent').innerHTML = h.items.length ? historyTable(h.items) :
    '<p class="muted">Not enough data yet.</p>';
  bindRowClicks();
}

/* Inline SVG bar chart. No chart library, and no bars without data. */
function seriesChart(series) {
  if (!series || !series.length) return notEnough('No dated scans recorded.');
  const byDay = {};
  series.forEach((r) => {
    byDay[r.day] = byDay[r.day] || { low: 0, suspicious: 0, high: 0, insufficient: 0 };
    byDay[r.day][r.risk_key] = r.count;
  });
  const days = Object.keys(byDay).map(Number).sort((a, b) => a - b);
  if (days.length < 2) {
    return notEnough(`Only ${days.length} day of activity recorded — a trend needs at least two.`);
  }
  const max = Math.max(...days.map((d) =>
    byDay[d].low + byDay[d].suspicious + byDay[d].high + byDay[d].insufficient));
  const W = 760, H = 180, pad = 28;
  const bw = Math.max(6, Math.min(40, (W - pad * 2) / days.length - 6));
  const bars = days.map((d, i) => {
    const x = pad + i * ((W - pad * 2) / days.length);
    const v = byDay[d];
    let y = H - pad;
    const segs = [['high', v.high], ['suspicious', v.suspicious],
      ['insufficient', v.insufficient], ['low', v.low]]
      .map(([k, n]) => {
        if (!n) return '';
        const h = (n / max) * (H - pad * 2);
        y -= h;
        return `<rect x="${x}" y="${y}" width="${bw}" height="${h}" class="bar-${k}"><title>${new Date(d * 1000).toLocaleDateString()} — ${k}: ${n}</title></rect>`;
      }).join('');
    return segs;
  }).join('');
  const labels = days.map((d, i) => {
    if (days.length > 8 && i % Math.ceil(days.length / 8) !== 0) return '';
    const x = pad + i * ((W - pad * 2) / days.length);
    return `<text x="${x}" y="${H - 8}" class="axis">${new Date(d * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img"
    aria-label="Scans per day by assessment">
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" class="axis-line"/>
    <text x="4" y="${pad}" class="axis">${max}</text>
    ${bars}${labels}</svg>
    <div class="legend"><span class="key key-low"></span>Low
    <span class="key key-suspicious"></span>Suspicious
    <span class="key key-insufficient"></span>Insufficient
    <span class="key key-high"></span>High</div>`;
}

/* ----------------------------------------------------------------- history */

function historyTable(items) {
  return `<div class="table-scroll"><table class="history">
    <thead><tr><th>Domain / URL</th><th>Assessment</th><th class="num">Score</th>
    <th>Source</th><th>Model</th><th>When</th></tr></thead><tbody>
    ${items.map((it) => `<tr class="row-click" data-id="${escapeHtml(it.id)}">
      <td class="hist-url" title="${escapeHtml(it.url)}">${escapeHtml(it.domain || it.url)}</td>
      <td><span class="pill pill-${escapeHtml(it.risk_key)}">${escapeHtml(it.risk_band)}</span></td>
      <td class="num">${escapeHtml(it.risk_score)}</td>
      <td class="muted-cell">${escapeHtml(it.source || 'Not available')}</td>
      <td class="muted-cell">${escapeHtml(it.model_version || 'Not available')}</td>
      <td class="muted-cell" title="${fmtTime(it.scanned_at)}">${relTime(it.scanned_at)}</td>
    </tr>`).join('')}</tbody></table></div>`;
}

let histState = { offset: 0, limit: 20 };

async function loadHistory() {
  const q = $('hist-q').value.trim();
  const band = $('hist-band').value;
  const [sort, order] = $('hist-sort').value.split(':');
  const params = new URLSearchParams({
    limit: histState.limit, offset: histState.offset, sort, order,
  });
  if (q) params.set('q', q);
  if (band) params.set('band', band);
  const h = await call(`/api/history?${params}`);
  $('history-note').textContent = api.user
    ? `Scans recorded for ${api.user.email}. Only you can see these.`
    : 'You are not signed in. These are scans from this browser session only.';
  $('history-wrap').innerHTML = h.items.length ? historyTable(h.items)
    : notEnough('No scans match this filter yet.');
  bindRowClicks();
  const pages = Math.ceil(h.total / histState.limit);
  const page = Math.floor(histState.offset / histState.limit) + 1;
  $('history-pager').innerHTML = h.total > histState.limit
    ? `<button class="btn btn-ghost btn-sm" ${histState.offset === 0 ? 'disabled' : ''} id="prev-page">Previous</button>
       <span class="muted">Page ${page} of ${pages} · ${h.total} scans</span>
       <button class="btn btn-ghost btn-sm" ${page >= pages ? 'disabled' : ''} id="next-page">Next</button>` : '';
  const p = $('prev-page'), n = $('next-page');
  if (p) p.addEventListener('click', () => { histState.offset -= histState.limit; loadHistory(); });
  if (n) n.addEventListener('click', () => { histState.offset += histState.limit; loadHistory(); });
}

['hist-q', 'hist-band', 'hist-sort'].forEach((id) =>
  $(id).addEventListener('input', () => { histState.offset = 0; loadHistory(); }));

$('clear-history').addEventListener('click', async () => {
  await call('/api/history', { method: 'DELETE' });
  histState.offset = 0;
  loadHistory();
});

function bindRowClicks() {
  document.querySelectorAll('.row-click').forEach((r) =>
    r.addEventListener('click', () => openDrawer(r.dataset.id)));
}

async function openDrawer(id) {
  try {
    const it = await call(`/api/history/${encodeURIComponent(id)}`);
    $('drawer-title').textContent = it.domain || 'Scan detail';
    $('drawer-body').innerHTML = `
      <div class="pill pill-${escapeHtml(it.risk_key)} big">${escapeHtml(it.risk_band)} · ${escapeHtml(it.risk_score)}/100</div>
      <div class="kv-list">
        ${[['URL', it.url], ['Domain', it.domain], ['Risk score', it.risk_score],
           ['Model confidence', it.confidence != null ? `${it.confidence}%` : null],
           ['Model version', it.model_version], ['Dataset version', it.dataset_version],
           ['Model', it.model_name], ['Detection source', it.source],
           ['Scope', it.scope === 'hostname' ? 'Hostname only' : 'Full URL'],
           ['Detection time', it.detection_ms != null ? `${it.detection_ms} ms` : null],
           ['Scanned', fmtTime(it.scanned_at)]]
          .map(([k, v]) => `<div class="kv-row"><span class="kv-key">${k}</span>
            <span class="kv-val">${v == null || v === '' ? '<span class="unavailable">Not available</span>' : escapeHtml(v)}</span></div>`).join('')}
      </div>
      <h4 class="sub">Security signals recorded</h4>
      ${it.signals && it.signals.length
        ? `<ul class="sig-list">${it.signals.map((s) => `<li>⚠ ${escapeHtml(s)}</li>`).join('')}</ul>`
        : '<p class="muted">No risk signals were recorded for this scan.</p>'}
      <p class="card-note">These signals contributed to the assessment. They do
      not by themselves prove malicious activity.</p>
      <button class="btn btn-ghost btn-sm" id="rescan-btn">Re-run this scan</button>`;
    $('drawer').hidden = false;
    $('rescan-btn').addEventListener('click', () => {
      $('drawer').hidden = true;
      $('url-input').value = it.url;
      gotoView('scanner');
      scanForm.requestSubmit();
    });
  } catch (e) { /* 404 for someone else's scan: silently ignore */ }
}

$('drawer-close').addEventListener('click', () => { $('drawer').hidden = true; });
$('drawer').addEventListener('click', (e) => {
  if (e.target.id === 'drawer') $('drawer').hidden = true;
});

/* ----------------------------------------------------------------- threats */

async function loadThreats() {
  const h = await call('/api/history?limit=100');
  const flagged = h.items.filter((i) => i.risk_key !== 'low');
  $('threats-wrap').innerHTML = flagged.length ? historyTable(flagged)
    : notEnough('No scans have been assessed Suspicious or High risk yet.');
  bindRowClicks();
  const s = await call('/api/analytics');
  $('threat-domains').innerHTML = s.top_domains && s.top_domains.length
    ? `<div class="kv-list">${s.top_domains.map((d) => `<div class="kv-row">
        <span class="kv-key">${escapeHtml(d.domain)}</span>
        <span class="kv-val">${d.count} scan${d.count > 1 ? 's' : ''} · worst ${d.worst_score}/100</span></div>`).join('')}</div>`
    : notEnough('No domain has been flagged more than once yet.');
}

/* --------------------------------------------------------------- analytics */

function barList(obj, emptyMsg) {
  const entries = Object.entries(obj || {});
  if (!entries.length) return notEnough(emptyMsg);
  const max = Math.max(...entries.map(([, v]) => v));
  return `<div class="barlist">${entries.map(([k, v]) => `
    <div class="barlist-row"><span class="barlist-key">${escapeHtml(k)}</span>
    <span class="barlist-bar"><i style="width:${(v / max) * 100}%"></i></span>
    <span class="barlist-val">${v}</span></div>`).join('')}</div>`;
}

async function loadAnalytics() {
  const s = await call('/api/analytics');
  if (!s.sufficient_data) {
    ['an-distribution', 'an-series', 'an-source', 'an-model'].forEach((id) => {
      $(id).innerHTML = notEnough('Analytics appear once scans have been recorded.');
    });
    return;
  }
  $('an-distribution').innerHTML = barList(
    { 'Low risk': s.low, Suspicious: s.suspicious, 'High risk': s.high,
      'Insufficient evidence': s.insufficient },
    'No assessments recorded.');
  $('an-series').innerHTML = seriesChart(s.series);
  $('an-source').innerHTML = barList(s.by_source, 'No detection sources recorded.');
  $('an-model').innerHTML = barList(s.by_model_version, 'No model versions recorded.');
}

/* ----------------------------------------------------------------- profile */

async function loadProfile() {
  if (!api.user) {
    $('profile-info').innerHTML = notEnough('Sign in to view account details.');
    return;
  }
  const u = api.user;
  $('profile-info').innerHTML = [
    ['Name', u.name], ['Email', u.email],
    ['Account created', fmtTime(u.created_at)],
    ['Last sign-in', u.last_login_at ? fmtTime(u.last_login_at) : 'This session'],
  ].map(([k, v]) => `<div class="kv-row"><span class="kv-key">${k}</span>
    <span class="kv-val">${escapeHtml(v)}</span></div>`).join('');
  $('profile-name').value = u.name;
}

$('profile-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const d = await call('/api/auth/profile', {
      method: 'PATCH', body: { name: $('profile-name').value },
    });
    api.user = d.user;
    $('profile-msg').textContent = 'Name updated.';
    $('profile-msg').hidden = false;
    loadProfile();
    enterApp();
  } catch (err) {
    $('profile-msg').textContent = err.message;
    $('profile-msg').hidden = false;
  }
});

$('password-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const d = await call('/api/auth/password', {
      method: 'POST',
      body: { current_password: $('pw-current').value, new_password: $('pw-new').value },
    });
    $('password-msg').textContent = d.message;
    $('password-msg').hidden = false;
    $('pw-current').value = $('pw-new').value = '';
  } catch (err) {
    $('password-msg').textContent = err.message;
    $('password-msg').hidden = false;
  }
});

$('logout-btn').addEventListener('click', doLogout);
$('logout-all-btn').addEventListener('click', async () => {
  try {
    await call('/api/auth/logout', {
      method: 'POST', body: { refresh_token: api.refresh, all_devices: true },
    });
  } catch (e) { /* ignore */ }
  clearSession();
  $('app').hidden = true; $('auth-screen').hidden = false;
  showAuthView('login');
  authNotice('Signed out on all devices.');
});

/* ---------------------------------------------------------------- settings */

const PREF_LABELS = {
  warn_on_suspicious: ['Warn on Suspicious assessments',
    'Show a warning for scores in the 31–70 band, not only High risk.'],
  block_high_risk: ['Interrupt on High risk',
    'Show the blocking warning screen when a link scores 71 or above.'],
  dns_layer_enabled: ['Enable the DNS layer on Android',
    'Hostname-only assessment for all apps. Much less accurate than full-URL analysis.'],
  store_full_urls: ['Store full URLs in history',
    'When off, only the domain is kept for scans you run.'],
  notifications: ['Notifications', 'Allow the Android app to notify you about blocked lookups.'],
};

async function loadSettings() {
  if (!api.user) {
    $('prefs').innerHTML = notEnough('Sign in to manage security preferences.');
  } else {
    const p = api.user.preferences || {};
    $('prefs').innerHTML = Object.entries(PREF_LABELS).map(([k, [title, desc]]) => `
      <label class="pref"><input type="checkbox" data-pref="${k}" ${p[k] ? 'checked' : ''}>
        <span><strong>${escapeHtml(title)}</strong><em>${escapeHtml(desc)}</em></span></label>`).join('');
    document.querySelectorAll('[data-pref]').forEach((cb) =>
      cb.addEventListener('change', async () => {
        const prefs = {};
        document.querySelectorAll('[data-pref]').forEach((x) => {
          prefs[x.dataset.pref] = x.checked;
        });
        try {
          const d = await call('/api/auth/profile', { method: 'PATCH', body: { preferences: prefs } });
          api.user = d.user;
          $('prefs-msg').textContent = 'Preferences saved.';
          $('prefs-msg').hidden = false;
        } catch (err) {
          $('prefs-msg').textContent = err.message;
          $('prefs-msg').hidden = false;
        }
      }));
  }
  const m = window.__meta || {};
  $('model-info').innerHTML = [
    ['Detection model', m.model_version], ['Dataset', m.dataset_version],
    ['Selected algorithm', m.best_model_name], ['Features', m.n_features],
    ['Decision threshold', m.threshold != null ? `${Math.round(m.threshold * 100)}/100` : null],
    ['Models compared', (m.models || []).join(', ')],
  ].map(([k, v]) => `<div class="kv-row"><span class="kv-key">${k}</span>
    <span class="kv-val">${v == null || v === '' ? '<span class="unavailable">Not available</span>' : escapeHtml(v)}</span></div>`).join('');
}

/* -------------------------------------------------------------------- init */

(async function init() {
  const stored = loadStored();
  if (stored && stored.refresh) {
    api.refresh = stored.refresh;
    if (await tryRefresh()) { await enterApp(); return; }
  }
  $('auth-screen').hidden = false;
  showAuthView('login');
})();
