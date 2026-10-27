/* ==========================================================
   DETEKSI KDRT – App Logic (v4 – WebSocket + Single %)
   ========================================================== */

/* -------- STATE -------- */
const S = {
    view: 'monitoring',
    selId: null,
    monitoring: false,
    det: 'normal',       // 'normal' | 'kdrt'
    pct: 0,              // Single percentage 0-100
    editId: null,
    devs: [],
    hist: [],
    users: [],
    nDev: 1,
    nHist: 1,
    socket: null,
    connected: false,
    auth: { authenticated: false, username: '', role: 'guest' },
    camState: {},        // {ip: {pct, det, persons, interaction}} - per-camera detection state
    lastIp: null,        // Last viewed camera IP — keeps updating pct in background
    // Video mode
    videoId: null,
    videoFile: null,
    videoProcessing: false,
    videoDone: false,
    videoPct: 0,
    videoDet: 'normal',
    videoPersons: 0,
    firebaseMessaging: null,
    pushEnabled: false,
    soundReady: false,
};

/* -------- DOM -------- */
const q = s => document.querySelector(s);
const qa = s => document.querySelectorAll(s);

const D = {
    pgMon: q('#pg-mon'), pgHist: q('#pg-hist'), pgVid: q('#pg-vid'), pgUsers: q('#pg-users'),
    navMon: q('#nav-mon'), navHist: q('#nav-hist'), navVid: q('#nav-vid'), navUsers: q('#nav-users'),
    loginScreen: q('#login-screen'), loginUsername: q('#login-username'), loginPassword: q('#login-password'),
    btnLogin: q('#btn-login'), loginError: q('#login-error'), authUser: q('#auth-user'), btnLogout: q('#btn-logout'),
    feedBgIdle: q('#feed-bg-idle'), splitView: q('#split-view'),
    feedBgRaw: q('#feed-bg-raw'), feedBgAnno: q('#feed-bg-anno'),
    feedLive: q('#feed-live'),
    flRing: q('#fl-ring'), flName: q('#fl-name'),
    flRtsp: q('#fl-rtsp'), flTag: q('#fl-tag'),
    overlay: q('#feed-overlay'), feedLabel: q('#feed-label'),
    vStatus: q('#v-status'), vPct: q('#v-pct'),
    vMon: q('#v-mon'), vDesc: q('#v-desc'),
    clList: q('#cl-list'), btnAdd: q('#btn-add'),
    htBody: q('#ht-body'), htEmpty: q('#ht-empty'),
    userBody: q('#user-body'), userEmpty: q('#user-empty'), btnUserAdd: q('#btn-user-add'),
    userUsername: q('#user-username'), userPassword: q('#user-password'), userRole: q('#user-role'),
    userActive: q('#user-active'), btnUserSave: q('#btn-user-save'),
    fAn: q('#f-an'), fAi: q('#f-ai'), fAu: q('#f-au'), fAp: q('#f-ap'), fAport: q('#f-aport'), fApath: q('#f-apath'),
    fEn: q('#f-en'), fEi: q('#f-ei'), fEu: q('#f-eu'), fEp: q('#f-ep'), fEport: q('#f-eport'), fEpath: q('#f-epath'),
    miTitle: q('#mi-title'), miSrc: q('#mi-src'), miVideo: q('#mi-video'), miMeta: q('#mi-meta'),
    // Video mode
    vidDropzone: q('#vid-dropzone'), vidFileInput: q('#vid-file-input'),
    vdzTitle: q('#vdz-title'), vdzSub: q('#vdz-sub'),
    btnVidProcess: q('#btn-vid-process'), btnVidStop: q('#btn-vid-stop'),
    vidProgressWrap: q('#vid-progress-wrap'), vidPbFill: q('#vid-pb-fill'), vidPbLabel: q('#vid-pb-label'),
    vidVStatus: q('#vid-v-status'), vidVPct: q('#vid-v-pct'), vidVPersons: q('#vid-v-persons'), vidVDesc: q('#vid-v-desc'),
    vidFeedIdle: q('#vid-feed-idle'), vidSplitView: q('#vid-split-view'),
    vidBgRaw: q('#vid-bg-raw'), vidBgAnno: q('#vid-bg-anno'),
    btnEnablePush: q('#btn-enable-push'), alertAudio: q('#alert-audio'),
};

function applyAuthUI() {
    const authed = !!S.auth.authenticated;
    document.body.classList.toggle('is-guest', !authed);
    if (D.loginScreen) D.loginScreen.classList.toggle('hidden', authed);
    if (D.authUser) {
        D.authUser.textContent = authed ? `${S.auth.username} (${S.auth.role})` : '-';
    }
    const isAdmin = S.auth.role === 'admin';
    if (D.navVid) D.navVid.classList.toggle('hidden', authed && !isAdmin);
    if (D.navUsers) D.navUsers.classList.toggle('hidden', authed && !isAdmin);
    if ((S.view === 'video' || S.view === 'users') && !isAdmin) go('monitoring');
    renderHist();
    renderUsers();
}

async function checkAuth() {
    try {
        const res = await fetch('/auth/status');
        S.auth = await res.json();
        localStorage.removeItem('kdrt_auth_token');
        if (S.auth.authenticated && sessionStorage.getItem('kdrt_tab_authenticated') !== '1') {
            await fetch('/auth/logout', { method: 'POST' }).catch(() => {});
            S.auth = { authenticated: false, username: '', role: 'guest' };
        }
    } catch {
        S.auth = { authenticated: false, username: '', role: 'guest' };
    }
    applyAuthUI();
    if (S.auth.authenticated) initSocket();
}

async function login() {
    const username = D.loginUsername.value.trim();
    const password = D.loginPassword.value;
    D.loginError.textContent = '';
    try {
        const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
            D.loginError.textContent = data.error || 'Login failed';
            return;
        }
        localStorage.removeItem('kdrt_auth_token');
        sessionStorage.setItem('kdrt_tab_authenticated', '1');
        S.auth = { authenticated: true, username: data.username, role: data.role };
        applyAuthUI();
        initSocket();
        toast(`Logged in as ${data.role}`);
    } catch (err) {
        D.loginError.textContent = err.message;
    }
}

async function logout() {
    try {
        await fetch('/auth/logout', { method: 'POST' });
    } catch {}
    localStorage.removeItem('kdrt_auth_token');
    sessionStorage.removeItem('kdrt_tab_authenticated');
    if (S.socket) {
        S.socket.disconnect();
        S.socket = null;
    }
    S.connected = false;
    S.auth = { authenticated: false, username: '', role: 'guest' };
    if (D.loginUsername) D.loginUsername.value = '';
    if (D.loginPassword) D.loginPassword.value = '';
    if (D.loginError) D.loginError.textContent = '';
    applyAuthUI();
    if (D.loginUsername) D.loginUsername.focus();
}

if (D.btnLogin) D.btnLogin.onclick = login;
if (D.loginPassword) D.loginPassword.onkeydown = e => { if (e.key === 'Enter') login(); };
if (D.loginUsername) D.loginUsername.onkeydown = e => { if (e.key === 'Enter') D.loginPassword.focus(); };
if (D.btnLogout) D.btnLogout.onclick = logout;

/* -------- WEBSOCKET -------- */
function initSocket() {
    if (S.socket) return;
    if (typeof io === 'undefined') {
        console.warn('Socket.IO not loaded — running in offline mode');
        return;
    }

    // Connect to Flask backend
    S.socket = io(window.location.origin);

    S.socket.on('connect', () => {
        S.connected = true;
        console.log('[WS] Connected to server');
        toast('Terhubung ke server');
    });

    S.socket.on('disconnect', () => {
        S.connected = false;
        console.log('[WS] Disconnected');
    });

    S.socket.on('server_status', data => {
        console.log(`[WS] Server ready, ${data.models_loaded} models loaded`);
        // Auto-start detection for ALL saved CCTV cameras
        startAllStreams();
    });

    // Detection updates from backend (per-camera)
    S.socket.on('detection_update', data => {
        const ip = data.ip;
        // Store per-camera state
        S.camState[ip] = {
            pct: data.percentage || 0,
            det: data.is_alert ? 'kdrt' : 'normal',
            persons: data.persons || 0,
            interaction: data.interaction || 0,
        };
        // Update UI if this camera is selected OR is the last viewed camera
        const selDev = S.devs.find(d => d.id === S.selId);
        const isSelected = selDev && selDev.ip === ip;
        const isLastViewed = !S.selId && S.lastIp === ip;
        if (isSelected || isLastViewed) {
            S.pct = data.percentage || 0;
            S.det = data.is_alert ? 'kdrt' : 'normal';
            S.persons = data.persons || 0;
            S.interaction = data.interaction || 0;
        }
        refresh();
    });

    // Live video frame (fallback — MJPEG is primary)
    // S.socket.on('frame') removed — using MJPEG stream via <img src> instead

    // History update from Firebase
    S.socket.on('history_update', data => {
        S.hist.unshift({
            id: data.id,
            name: data.name,
            time: data.time,
            date: data.date,
            url: data.url,
            evidenceVideoUrl: data.evidence_video_url || data.evidence_video_local_url || '',
            evidenceVideoLocalUrl: data.evidence_video_local_url || '',
            evidenceVideoDownloadUrl: data.evidence_video_download_url || '',
            evidenceVideoName: data.evidence_video_filename || '',
            evidenceVideoDuration: data.evidence_video_duration_seconds || 0,
            evidenceStatus: data.evidence_status || 'processing',
            probability: data.probability,
        });
        if (S.view === 'history') renderHist();
        notifyAlert('KDRT detected - evidence video is being saved', {
            body: `${data.camera_name || 'Camera'} - evidence video is being saved`,
            tag: data.id || 'kdrt-alert',
        });
    });

    // Full history list from Firebase
    S.socket.on('history_list', data => {
        S.hist = data.map(d => ({
            id: d.id,
            name: d.name,
            time: d.time,
            date: d.date,
            url: d.url,
            evidenceVideoUrl: d.evidence_video_url || d.evidence_video_local_url || '',
            evidenceVideoLocalUrl: d.evidence_video_local_url || '',
            evidenceVideoDownloadUrl: d.evidence_video_download_url || '',
            evidenceVideoName: d.evidence_video_filename || '',
            evidenceVideoDuration: d.evidence_video_duration_seconds || 0,
            evidenceStatus: d.evidence_status || '',
            probability: d.probability,
        }));
        renderHist();
    });

    S.socket.on('history_evidence_update', data => {
        const item = S.hist.find(h => h.id === data.id);
        if (!item) return;
        item.evidenceVideoUrl = data.evidence_video_url || data.evidence_video_local_url || item.evidenceVideoUrl || '';
        item.evidenceVideoLocalUrl = data.evidence_video_local_url || item.evidenceVideoLocalUrl || '';
        item.evidenceVideoDownloadUrl = data.evidence_video_download_url || item.evidenceVideoDownloadUrl || '';
        item.evidenceVideoName = data.evidence_video_filename || item.evidenceVideoName || '';
        item.evidenceVideoDuration = data.evidence_video_duration_seconds || item.evidenceVideoDuration || 0;
        item.evidenceStatus = data.evidence_status || 'ready';
        if (S.view === 'history') renderHist();
        toast('Evidence video ready in history');
    });

    // History deleted
    S.socket.on('history_deleted', data => {
        S.hist = S.hist.filter(h => h.id !== data.id);
        renderHist();
        toast('History successfully deleted');
    });

    S.socket.on('history_delete_denied', data => {
        toast(data.error || 'Only an admin can delete history');
    });

    // CCTV connection confirmed → set online (green)
    S.socket.on('cctv_connected', data => {
        const dev = S.devs.find(d => d.ip === data.ip);
        if (dev) { dev.online = true; renderCctv(); }
        toast(data.name + ' terhubung');
    });

    // Connection errors → set offline (red)
    S.socket.on('connection_error', data => {
        const dev = S.devs.find(d => d.ip === data.ip);
        if (dev) { dev.online = false; renderCctv(); }
        toast('Error: ' + data.error);
    });

    // Stream closed → set offline (red) but DON'T stop monitoring state
    S.socket.on('stream_closed', data => {
        const ip = data && data.ip;
        if (ip) {
            const dev = S.devs.find(d => d.ip === ip);
            if (dev) { dev.online = false; }
            // Clear per-camera state
            delete S.camState[ip];
            // Reset UI if this camera was selected
            const selDev = S.devs.find(d => d.id === S.selId);
            if (selDev && selDev.ip === ip) {
                S.monitoring = false;
                S.pct = 0;
                S.det = 'normal';
                D.feedBg.src = 'assets/Background %231.png';
            }
        }
        refresh();
    });
}

/* -------- AUTO-START ALL STREAMS -------- */
function startAllStreams() {
    if (!S.socket || !S.connected) return;
    const cameras = S.devs.map(d => ({
        ip: d.ip, name: d.name,
        username: d.username || '', password: d.password || '',
        port: d.port || '554',
        path: d.path || '/Streaming/Channels/101'
    }));
    if (cameras.length > 0) {
        S.socket.emit('start_all_streams', { cameras });
        console.log(`[WS] Auto-starting ${cameras.length} camera streams`);
    }
}

/* -------- NAV -------- */
function go(v) {
    S.view = v;
    D.pgMon.classList.toggle('active', v === 'monitoring');
    D.pgHist.classList.toggle('active', v === 'history');
    D.pgVid.classList.toggle('active', v === 'video');
    if (D.pgUsers) D.pgUsers.classList.toggle('active', v === 'users');
    D.navMon.classList.toggle('active', v === 'monitoring');
    D.navHist.classList.toggle('active', v === 'history');
    D.navVid.classList.toggle('active', v === 'video');
    if (D.navUsers) D.navUsers.classList.toggle('active', v === 'users');
    if (v === 'history') {
        if (S.socket && S.connected) S.socket.emit('get_history');
        renderHist();
    }
    if (v === 'users') fetchUsers();
    if (v !== 'video' && S.videoProcessing && S.videoId) {
        // Stop video when leaving video mode
        stopVideo();
    }
}
D.navMon.onclick = () => go('monitoring');
D.navHist.onclick = () => go('history');
D.navVid.onclick  = () => go('video');
if (D.navUsers) D.navUsers.onclick = () => go('users');

/* -------- CCTV LIST -------- */
function renderCctv() {
    D.clList.innerHTML = '';
    if (!S.devs.length) {
        D.clList.innerHTML = `
        <div class="cl-empty">
            <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.4">
                <rect x="6" y="14" width="26" height="18" rx="3"/><circle cx="19" cy="23" r="5.5"/>
                <polygon points="32,18 42,13 42,33 32,28"/><line x1="4" y1="44" x2="44" y2="4" stroke-width="2.2"/>
            </svg>
            <span>No CCTV yet.<br>Press <b>+</b> to add one.</span>
        </div>`;
        return;
    }
    S.devs.forEach(d => {
        const el = document.createElement('div');
        el.className = 'cl-card';
        if (S.selId === d.id) {
            el.classList.add('sel');
            if (S.det === 'kdrt') el.classList.add('is-red');
        }
        el.innerHTML = `
            <img src="assets/Frame ${d.online ? 3 : 5}.png" alt="" class="cl-ico">
            <div class="cl-info"><div class="cl-name">${esc(d.name)}</div><div class="cl-ip">IP Address: ${esc(d.ip)}</div></div>
            <div class="cl-dot ${d.online ? 'on' : 'off'}"></div>
            <button class="cl-edit" data-id="${d.id}" title="Edit">
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="10" cy="4" r="1.2"/><circle cx="10" cy="10" r="1.2"/><circle cx="10" cy="16" r="1.2"/></svg>
            </button>`;
        el.addEventListener('click', e => { if (!e.target.closest('.cl-edit')) selectDev(d.id) });
        el.querySelector('.cl-edit').addEventListener('click', e => { e.stopPropagation(); openEdit(d.id) });
        D.clList.appendChild(el);
    });
}

/* -------- SELECT / DISPLAY -------- */
function selectDev(id) {
    const dev = S.devs.find(d => d.id === id);

    if (S.selId === id) {
        // Deselect → stop VIEWING only (detection continues in background)
        S.lastIp = dev ? dev.ip : null;  // Remember which camera to track
        S.selId = null;
        S.monitoring = false;
        // DON'T reset S.pct and S.det — keep showing background data
        D.feedBgIdle.src = 'assets/Background %231.png';  // Reset video to placeholder
        D.feedBgRaw.src = '';
        D.feedBgAnno.src = '';
        // NOTE: We do NOT emit disconnect_cctv — detection keeps running!
    } else {
        // Select → start VIEWING (detection already running in background)
        S.selId = id;
        S.monitoring = true;
        S.lastIp = dev ? dev.ip : null;  // Track current camera

        // Restore per-camera detection state if available
        if (dev && S.camState[dev.ip]) {
            const cs = S.camState[dev.ip];
            S.pct = cs.pct;
            S.det = cs.det;
            S.persons = cs.persons;
            S.interaction = cs.interaction;
        }

        if (dev) {
            // Use MJPEG stream for smooth video
            setTimeout(() => {
                D.feedBgRaw.src = `/video_feed/raw/${dev.ip}?t=${Date.now()}`;
                D.feedBgAnno.src = `/video_feed/anno/${dev.ip}?t=${Date.now()}`;
                D.feedBgIdle.classList.add('hidden');
                D.splitView.classList.remove('hidden');
            }, 300);
        }
    }
    refresh();
}

function refresh() {
    const dev = S.devs.find(d => d.id === S.selId);
    const lastDev = S.lastIp ? S.devs.find(d => d.ip === S.lastIp) : null;
    const k = S.det === 'kdrt';

    if (S.monitoring && dev) {
        D.feedBgIdle.classList.add('hidden');
        D.splitView.classList.remove('hidden');
        D.feedLive.classList.add('hidden');  // Hide center overlay — show raw video
        D.overlay.classList.remove('hidden');
        D.flName.textContent = dev.name;
        D.flRtsp.textContent = `rtsp://${dev.ip}:${dev.port || '554'}${dev.path || ''}`;
        D.feedLabel.textContent = `${dev.name} — ${dev.ip}`;
        D.flRing.className = 'fl-ring' + (k ? ' is-red' : '');
        D.flTag.className = 'fl-tag' + (k ? ' is-red' : '');
        D.flTag.innerHTML = `<span class="fl-dot"></span>${k ? 'KDRT Detected' : 'Normal'}`;
        D.vMon.textContent = 'Active'; D.vMon.className = 'cs-val green';
    } else {
        D.splitView.classList.add('hidden');
        D.feedBgIdle.classList.remove('hidden');
        D.feedBgIdle.style.opacity = '0.4';
        D.feedLive.classList.add('hidden');
        D.overlay.classList.add('hidden');
        D.vMon.textContent = 'Background'; D.vMon.className = 'cs-val yellow';
    }

    // Aggregate background data from all cameras
    const camIps = Object.keys(S.camState);
    const activeCount = camIps.length;
    let totalPersons = 0;
    let anyKdrt = false;
    const camDetails = [];
    camIps.forEach(ip => {
        const cs = S.camState[ip];
        const d = S.devs.find(x => x.ip === ip);
        totalPersons += cs.persons || 0;
        if (cs.det === 'kdrt') anyKdrt = true;
        if (d) camDetails.push({ name: d.name, persons: cs.persons || 0, det: cs.det });
    });

    // Status
    if (S.monitoring) {
        D.vStatus.textContent = k ? 'KDRT' : 'Normal';
        D.vStatus.className = 'cs-val ' + (k ? 'red' : 'green');
    } else if (activeCount > 0) {
        D.vStatus.textContent = anyKdrt ? 'KDRT Detected!' : `${activeCount} Active Cameras`;
        D.vStatus.className = 'cs-val ' + (anyKdrt ? 'red' : 'green');
    } else {
        D.vStatus.textContent = 'Waiting...';
        D.vStatus.className = 'cs-val yellow';
    }

    // Percentage — always shows (even in background mode)
    D.vPct.textContent = S.pct + '%';
    D.vPct.className = 'cs-val ' + (S.pct > 70 ? 'red' : 'green');

    // Person count: use selected camera's count, or total from all cameras
    const displayPersons = S.monitoring ? (S.persons || 0) : totalPersons;
    const pInfo = displayPersons ? ` (${displayPersons} people detected)` : '';

    // Background camera info
    let bgLabel = '';
    if (!S.monitoring && activeCount > 0) {
        const lines = camDetails.map(c =>
            `${c.name}: ${c.persons} people${c.det === 'kdrt' ? ' ⚠️KDRT' : ''}`
        ).join(' · ');
        bgLabel = `<br><em style="font-size:0.85em;color:#888">${lines}</em>`;
    } else if (!S.monitoring) {
        bgLabel = '<br><em style="font-size:0.85em;color:#888">Waiting for CCTV connection...</em>';
    }

    D.vDesc.innerHTML = k
        ? `"Current condition detected as <strong class="red">KDRT</strong><br>evidence video will be saved to History"`
        : `"Current condition is <strong class="green">Normal</strong>${pInfo}${bgLabel}"`;

    renderCctv();
}

/* -------- HISTORY -------- */
function renderHist() {
    if (!S.hist.length) { D.htBody.innerHTML = ''; D.htEmpty.classList.remove('hidden'); return; }
    D.htEmpty.classList.add('hidden');
    D.htBody.innerHTML = '';
    const canDeleteHistory = S.auth.role === 'admin';
    S.hist.forEach(h => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
        <td>${esc(h.name)}</td><td>${esc(h.time)}</td><td>${esc(h.date)}</td>
        <td><div class="ha">
            <button class="ha-btn ha-view" title="View evidence video"><svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 11s4-7 10-7 10 7 10 7-4 7-10 7S1 11 1 11z"/><circle cx="11" cy="11" r="3"/></svg></button>
            <button class="ha-btn ha-dl" title="Download evidence video"><svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M19 14v3a2 2 0 01-2 2H5a2 2 0 01-2-2v-3"/><polyline points="7,10 11,14 15,10"/><line x1="11" y1="14" x2="11" y2="3"/></svg></button>
            ${canDeleteHistory ? '<button class="ha-btn ha-del" title="Delete"><svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3,6 5,6 19,6"/><path d="M17 6v12a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h2a2 2 0 012 2v2"/></svg></button>' : ''}
        </div></td>`;
        tr.querySelector('.ha-view').onclick = () => {
            D.miTitle.textContent = h.name;
            const playbackUrl = h.evidenceVideoLocalUrl || h.evidenceVideoUrl;
            if (playbackUrl) {
                D.miSrc.classList.add('hidden');
                D.miSrc.src = '';
                D.miVideo.classList.remove('hidden');
                D.miVideo.setAttribute('controls', 'controls');
                D.miVideo.setAttribute('preload', 'metadata');
                D.miVideo.src = playbackUrl;
                D.miVideo.onerror = () => {
                    D.miMeta.textContent = 'Evidence video cannot be played by the browser. Use the download button.';
                };
                D.miVideo.load();
            } else {
                D.miVideo.pause();
                D.miVideo.classList.add('hidden');
                D.miVideo.src = '';
                D.miSrc.classList.remove('hidden');
                D.miSrc.src = h.url || 'assets/Background %231.png';
            }
            const evidenceInfo = playbackUrl
                ? ` | Evidence video: ${h.evidenceVideoDuration ? Math.round(h.evidenceVideoDuration) + ' seconds' : 'available'}`
                : ' | Evidence video: processing';
            D.miMeta.textContent = h.date + ' — ' + h.time + evidenceInfo;
            modal('m-img', true);
        };
        tr.querySelector('.ha-dl').onclick = () => {
            const downloadUrl = h.evidenceVideoDownloadUrl || h.evidenceVideoLocalUrl || h.evidenceVideoUrl || h.url;
            if (downloadUrl) {
                const a = document.createElement('a');
                a.href = downloadUrl; a.download = h.evidenceVideoName || h.name; a.target = '_blank';
                a.click();
            } else {
                toast('File not available');
            }
        };
        const delBtn = tr.querySelector('.ha-del');
        if (delBtn) {
            delBtn.onclick = () => {
                if (S.auth.role !== 'admin') return toast('Only an admin can delete history');
                if (S.socket && S.connected) {
                    S.socket.emit('delete_history', { id: h.id });
                } else {
                    toast('Cannot delete while offline');
                }
            };
        }
        D.htBody.appendChild(tr);
    });
}

/* -------- USER MANAGEMENT -------- */
async function fetchUsers() {
    if (S.auth.role !== 'admin') return;
    try {
        const res = await fetch('/users');
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to read users');
        S.users = data.users || [];
        renderUsers();
    } catch (err) {
        toast('Error user: ' + err.message);
    }
}

function renderUsers() {
    if (!D.userBody) return;
    if (S.auth.role !== 'admin') {
        D.userBody.innerHTML = '';
        if (D.userEmpty) D.userEmpty.classList.add('hidden');
        return;
    }
    if (!S.users.length) {
        D.userBody.innerHTML = '';
        if (D.userEmpty) D.userEmpty.classList.remove('hidden');
        return;
    }
    if (D.userEmpty) D.userEmpty.classList.add('hidden');
    D.userBody.innerHTML = '';
    S.users.forEach(u => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${esc(u.username)}</td>
            <td>
                <select class="user-role-select">
                    <option value="user" ${u.role === 'user' ? 'selected' : ''}>user</option>
                    <option value="admin" ${u.role === 'admin' ? 'selected' : ''}>admin</option>
                </select>
            </td>
            <td><button class="dbtn ghost user-toggle">${u.active ? 'Active' : 'Inactive'}</button></td>
            <td><div class="ha">
                <button class="ha-btn user-pass" title="Reset password"><svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="10" width="16" height="9" rx="2"/><path d="M7 10V7a4 4 0 018 0v3"/></svg></button>
                <button class="ha-btn ha-del user-delete" title="Delete user"><svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3,6 5,6 19,6"/><path d="M17 6v12a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h2a2 2 0 012 2v2"/></svg></button>
            </div></td>`;
        tr.querySelector('.user-role-select').onchange = e => updateUser(u.id, { role: e.target.value });
        tr.querySelector('.user-toggle').onclick = () => updateUser(u.id, { active: !u.active });
        tr.querySelector('.user-pass').onclick = () => {
            const password = prompt('New password for ' + u.username);
            if (password) updateUser(u.id, { password });
        };
        tr.querySelector('.user-delete').onclick = () => {
            if (confirm('Delete user ' + u.username + '?')) deleteUser(u.id);
        };
        D.userBody.appendChild(tr);
    });
}

async function createUser() {
    const payload = {
        username: D.userUsername.value.trim(),
        password: D.userPassword.value,
        role: D.userRole.value,
        active: D.userActive.checked,
    };
    try {
        const res = await fetch('/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to add user');
        modal('m-user', false);
        D.userUsername.value = '';
        D.userPassword.value = '';
        D.userRole.value = 'user';
        D.userActive.checked = true;
        toast('User successfully added');
        fetchUsers();
    } catch (err) {
        toast('Error user: ' + err.message);
    }
}

async function updateUser(id, payload) {
    try {
        const res = await fetch('/users/' + encodeURIComponent(id), {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to update user');
        toast('User successfully updated');
        fetchUsers();
    } catch (err) {
        toast('Error user: ' + err.message);
        fetchUsers();
    }
}

async function deleteUser(id) {
    try {
        const res = await fetch('/users/' + encodeURIComponent(id), { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to delete user');
        toast('User successfully deleted');
        fetchUsers();
    } catch (err) {
        toast('Error user: ' + err.message);
    }
}

if (D.btnUserAdd) D.btnUserAdd.onclick = () => modal('m-user', true);
if (D.btnUserSave) D.btnUserSave.onclick = createUser;

/* -------- CRUD -------- */
function handleIpChange(ipVal, elements) {
    const isWebcam = /^\d+$/.test(ipVal.trim());
    elements.forEach(el => {
        el.disabled = isWebcam;
        el.parentElement.style.opacity = isWebcam ? '0.4' : '1';
        if (isWebcam) el.value = '-';
        else if (el.value === '-') el.value = '';
    });
}
D.fAi.addEventListener('input', e => handleIpChange(e.target.value, [D.fAu, D.fAp, D.fAport, D.fApath]));
D.fEi.addEventListener('input', e => handleIpChange(e.target.value, [D.fEu, D.fEp, D.fEport, D.fEpath]));

D.btnAdd.onclick = () => {
    D.fAn.value = ''; D.fAi.value = ''; D.fAu.value = ''; D.fAp.value = ''; D.fAport.value = ''; D.fApath.value = '';
    [D.fAu, D.fAp, D.fAport, D.fApath].forEach(el => { el.disabled = false; el.parentElement.style.opacity = '1'; });
    modal('m-add', true); setTimeout(() => D.fAn.focus(), 180);
};
q('#btn-save-add').onclick = () => {
    const nm = D.fAn.value.trim(), ip = D.fAi.value.trim();
    const isWebcam = /^\d+$/.test(ip);
    const user = isWebcam ? '' : D.fAu.value.trim(), pass = isWebcam ? '' : D.fAp.value.trim();
    const port = isWebcam ? '' : (D.fAport.value.trim() || '554');
    const path = isWebcam ? '' : (D.fApath.value.trim() || '/Streaming/Channels/101');
    if (!nm || !ip) return toast('Name and IP Address are required');
    // Check duplicate IP
    if (S.devs.some(d => d.ip === ip)) return toast('IP Address already registered');
    const newDev = { id: S.nDev++, name: nm, ip, username: user, password: pass, port, path, online: false };
    S.devs.push(newDev);
    modal('m-add', false); renderCctv(); toast(nm + ' successfully added');
    // Auto-start detection for the new camera
    if (S.socket && S.connected) {
        S.socket.emit('connect_cctv', {
            ip, name: nm, username: user, password: pass, port, path
        });
    }
};

function openEdit(id) {
    const d = S.devs.find(x => x.id === id); if (!d) return;
    S.editId = id;
    D.fEn.value = d.name; D.fEi.value = d.ip;
    D.fEu.value = d.username || ''; D.fEp.value = d.password || '';
    D.fEport.value = d.port || '554'; D.fEpath.value = d.path || '/Streaming/Channels/101';
    handleIpChange(d.ip, [D.fEu, D.fEp, D.fEport, D.fEpath]);
    modal('m-edit', true); setTimeout(() => D.fEn.focus(), 180);
}
q('#btn-save-edit').onclick = () => {
    const nm = D.fEn.value.trim(), ip = D.fEi.value.trim();
    const isWebcam = /^\d+$/.test(ip);
    const user = isWebcam ? '' : D.fEu.value.trim(), pass = isWebcam ? '' : D.fEp.value.trim();
    const port = isWebcam ? '' : (D.fEport.value.trim() || '554');
    const path = isWebcam ? '' : (D.fEpath.value.trim() || '/Streaming/Channels/101');
    if (!nm || !ip) return toast('Name and IP Address are required');
    const d = S.devs.find(x => x.id === S.editId);
    if (d) { d.name = nm; d.ip = ip; d.username = user; d.password = pass; d.port = port; d.path = path; }
    modal('m-edit', false); refresh(); toast(nm + ' successfully updated');
};
q('#btn-del-cctv').onclick = () => {
    const d = S.devs.find(x => x.id === S.editId); if (!d) return;
    // Always stop background detection for this camera
    if (S.socket && S.connected) S.socket.emit('disconnect_cctv', { ip: d.ip });
    delete S.camState[d.ip];
    if (S.selId === S.editId) {
        S.selId = null; S.monitoring = false; S.pct = 0; S.det = 'normal';
        D.feedBg.src = 'assets/Background %231.png';
    }
    S.devs = S.devs.filter(x => x.id !== S.editId);
    modal('m-edit', false); refresh(); toast(d.name + ' successfully deleted');
};

/* -------- MODALS -------- */
function stopEvidenceVideo() {
    if (!D.miVideo) return;
    D.miVideo.pause();
    D.miVideo.removeAttribute('src');
    D.miVideo.load();
}
function modal(id, show) {
    q('#' + id).classList.toggle('hidden', !show);
    if (id === 'm-img' && !show) stopEvidenceVideo();
}
qa('[data-close]').forEach(b => b.onclick = () => modal(b.dataset.close, false));
qa('.overlay').forEach(o => o.addEventListener('click', e => {
    if (e.target === o) {
        if (o.id === 'm-img') stopEvidenceVideo();
        o.classList.add('hidden');
    }
}));
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        stopEvidenceVideo();
        qa('.overlay').forEach(m => m.classList.add('hidden'));
    }
});
D.fAn.onkeydown = e => { if (e.key === 'Enter') D.fAi.focus() };
D.fAi.onkeydown = e => { if (e.key === 'Enter') q('#btn-save-add').click() };
D.fEn.onkeydown = e => { if (e.key === 'Enter') D.fEi.focus() };
D.fEi.onkeydown = e => { if (e.key === 'Enter') q('#btn-save-edit').click() };

/* -------- TOAST -------- */
function toast(msg) {
    const box = q('#toast-box');
    const t = document.createElement('div');
    const now = new Date();
    const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const isAlert = /KDRT|bukti|Video/i.test(msg);
    t.className = 'toast-msg';
    t.innerHTML = `
        <div class="toast-head">
            <span class="toast-app-dot">!</span>
            <span>KDRT Monitor</span>
            <span class="toast-time">${esc(time)}</span>
        </div>
        <div class="toast-body">
            <div class="toast-icon">${isAlert ? '!' : 'i'}</div>
            <div>
                <div class="toast-title">${isAlert ? 'System Alert' : 'System Status'}</div>
                <div class="toast-text">${esc(msg)}</div>
            </div>
        </div>`;
    box.appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));
    setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 250) }, isAlert ? 5200 : 3200);
}

/* -------- UTIL -------- */
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML }
function pad(n) { return String(n).padStart(2, '0') }

/* -------- BROWSER NOTIFICATION + SOUND -------- */
let audioCtx = null;

function unlockNotificationSound() {
    if (S.soundReady) return;
    try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return;
        audioCtx = audioCtx || new Ctx();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        S.soundReady = true;
    } catch (err) {
        console.warn('[Audio] Unable to unlock notification sound', err);
    }
}

function playAlertSound() {
    try {
        unlockNotificationSound();
        if (!audioCtx) return;
        if (audioCtx.state === 'suspended') {
            audioCtx.resume().then(playAlertSound).catch(err => {
                console.warn('[Audio] Resume failed', err);
            });
            return;
        }
        if (audioCtx.state !== 'running') return;
        const now = audioCtx.currentTime + 0.02;
        [0, 0.18].forEach(offset => {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, now + offset);
            gain.gain.setValueAtTime(0.0001, now + offset);
            gain.gain.exponentialRampToValueAtTime(0.18, now + offset + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.14);
            osc.connect(gain).connect(audioCtx.destination);
            osc.start(now + offset);
            osc.stop(now + offset + 0.16);
        });
    } catch (err) {
        console.warn('[Audio] Notification sound failed', err);
    }
}

function showWindowsNotification(title, body, tag = 'kdrt-alert') {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    try {
        if (navigator.serviceWorker?.ready) {
            navigator.serviceWorker.ready.then(registration => {
                registration.showNotification(title, {
                    body,
                    icon: '/assets/Frame 9.png',
                    badge: '/assets/Frame 9.png',
                    tag,
                    requireInteraction: true,
                    data: { url: '/' },
                });
            }).catch(() => {
                new Notification(title, { body, icon: '/assets/Frame 9.png', tag });
            });
        } else {
            new Notification(title, { body, icon: '/assets/Frame 9.png', tag });
        }
    } catch (err) {
        console.warn('[Notification] Windows notification failed', err);
    }
}

function notifyAlert(message, options = {}) {
    const title = options.title || 'KDRT Alert Detected';
    const body = options.body || message || 'The system detected indications of violence.';
    const tag = options.tag || 'kdrt-alert';
    toast(message || `${title} - ${body}`);
    playAlertSound();
    if (document.hidden || !document.hasFocus()) {
        showWindowsNotification(title, body, tag);
    }
}

async function fetchMessagingConfig() {
    const res = await fetch('/firebase_messaging_config');
    if (!res.ok) throw new Error('Failed to read Firebase Messaging configuration');
    return res.json();
}

async function registerFcmToken(token) {
    const res = await fetch('/register_fcm_token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to save FCM token');
}

async function registerWebPushSubscription(subscription) {
    const res = await fetch('/register_web_push_subscription', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subscription: subscription.toJSON() }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to save Web Push subscription');
}

function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

async function verifyStandardPushSubscription(registration, vapidKey) {
    const applicationServerKey = urlBase64ToUint8Array(vapidKey);
    let subscription = await registration.pushManager.getSubscription();
    if (subscription) {
        const currentKey = subscription.options?.applicationServerKey;
        if (currentKey && currentKey.byteLength && currentKey.byteLength !== applicationServerKey.byteLength) {
            await subscription.unsubscribe();
            subscription = null;
        }
    }
    if (!subscription) {
        subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey,
        });
    }
    return subscription;
}

function setPushButtonState(text, active) {
    if (!D.btnEnablePush) return;
    D.btnEnablePush.classList.toggle('push-active', !!active);
    D.btnEnablePush.title = text || 'Notification settings';
    D.btnEnablePush.setAttribute('aria-label', text || 'Notification settings');
    if (active) localStorage.setItem('kdrt_push_active', '1');
    else localStorage.removeItem('kdrt_push_active');
}

async function resetLocalServiceWorkers() {
    if (!navigator.serviceWorker?.getRegistrations) return;
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map(registration => {
        const sameOrigin = registration.scope && registration.scope.startsWith(window.location.origin + '/');
        return sameOrigin ? registration.unregister() : Promise.resolve(false);
    }));
}

async function enableBrowserNotifications(showToast = true) {
    try {
        unlockNotificationSound();
        if (!('Notification' in window) || !('serviceWorker' in navigator)) {
            if (showToast) toast('Browser does not support background notifications');
            return false;
        }
        if (typeof firebase === 'undefined' || !firebase.messaging) {
            if (showToast) toast('Firebase Messaging SDK not yet available');
            return false;
        }

        const cfg = await fetchMessagingConfig();
        if (!cfg.enabled) {
            if (showToast) toast('FCM not yet active: ' + (cfg.reason || 'configuration incomplete'));
            return false;
        }

        const permission = Notification.permission === 'granted'
            ? 'granted'
            : await Notification.requestPermission();
        if (permission !== 'granted') {
            if (showToast) toast('Notification permission not granted');
            return false;
        }

        await resetLocalServiceWorkers();
        const swUrl = `/firebase-messaging-sw.js?v=${Date.now()}`;
        const registration = await navigator.serviceWorker.register(swUrl, {
            updateViaCache: 'none',
        });
        await navigator.serviceWorker.ready;
        let standardSubscription = null;
        let pushSubscribeError = null;
        try {
            standardSubscription = await verifyStandardPushSubscription(registration, cfg.vapidKey);
            await registerWebPushSubscription(standardSubscription);
        } catch (pushErr) {
            pushSubscribeError = pushErr;
            console.warn('[PushManager] Standard subscription failed', pushErr);
        }
        if (!firebase.apps.length) firebase.initializeApp(cfg.config);
        const messaging = firebase.messaging();
        let token = '';
        try {
            token = await messaging.getToken({
                vapidKey: cfg.vapidKey,
                serviceWorkerRegistration: registration,
            });
        } catch (tokenErr) {
            const pushMsg = pushSubscribeError ? ` PushManager: ${pushSubscribeError.message}.` : '';
            throw new Error(
                'Failed to create FCM token.' + pushMsg +
                ' Check the connection to the Google/FCM push service, Chrome notification permission, or try another Chrome browser/profile.'
            );
        }
        if (!token) {
            if (showToast) toast('Notification token not yet available');
            return false;
        }

        await registerFcmToken(token);
        messaging.onMessage(payload => {
            const title = payload.notification?.title || 'KDRT Alert Detected';
            const body = payload.notification?.body || 'The system detected indications of violence.';
            notifyAlert(`${title} - ${body}`, {
                title,
                body,
                tag: payload.data?.history_id || 'kdrt-alert',
            });
        });

        S.firebaseMessaging = messaging;
        S.pushEnabled = true;
        setPushButtonState('Notifications Active', true);
        if (showToast) {
            toast('Browser notifications active');
            playAlertSound();
        }
        return true;
    } catch (err) {
        console.error('[FCM] Enable failed', err);
        setPushButtonState('Notifications', false);
        if (showToast) toast('Failed to enable notifications: ' + err.message);
        return false;
    }
}

if (D.btnEnablePush) {
    D.btnEnablePush.onclick = () => enableBrowserNotifications(true);
    if (localStorage.getItem('kdrt_push_active') === '1') {
        setPushButtonState('Notifications Active', true);
    }
}
window.addEventListener('pointerdown', unlockNotificationSound, { once: true });

/* -------- INIT -------- */
renderCctv(); refresh(); renderHist(); checkAuth();
if ('Notification' in window && Notification.permission === 'granted') {
    enableBrowserNotifications(false);
}

/* ======================================================
   VIDEO MODE
   ====================================================== */

/* -------- Drag & Drop / File select -------- */
function setVideoFile(file) {
    if (!file) return;
    S.videoFile = file;
    D.vdzTitle.textContent = file.name;
    D.vdzSub.textContent = (file.size / 1024 / 1024).toFixed(1) + ' MB';
    D.btnVidProcess.disabled = false;
    D.vidDropzone.classList.add('has-file');
    toast('Video dipilih: ' + file.name);
}

D.vidDropzone.onclick = () => { if (!S.videoProcessing) D.vidFileInput.click(); };
D.vidFileInput.onchange = e => { if (e.target.files[0]) setVideoFile(e.target.files[0]); };

D.vidDropzone.addEventListener('dragover', e => {
    e.preventDefault();
    if (!S.videoProcessing) D.vidDropzone.classList.add('drag-over');
});
D.vidDropzone.addEventListener('dragleave', () => D.vidDropzone.classList.remove('drag-over'));
D.vidDropzone.addEventListener('drop', e => {
    e.preventDefault();
    D.vidDropzone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) setVideoFile(f);
    else toast('File bukan video');
});

/* -------- Upload & Process -------- */
D.btnVidProcess.onclick = async () => {
    if (!S.videoFile) return toast('Select a video first');
    if (S.videoProcessing) return;

    // Upload step
    D.btnVidProcess.disabled = true;
    D.btnVidProcess.innerHTML = '<svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="9" stroke-dasharray="4 2"/></svg> Uploading...';
    toast('Uploading video...');

    const fd = new FormData();
    fd.append('video', S.videoFile);

    try {
        const res  = await fetch('/upload_video', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) { toast('Upload failed: ' + (data.error || res.statusText)); resetVidUI(); return; }

        S.videoId = data.video_id;
        toast('Upload successful, starting processing...');

        // Start processing
        const res2  = await fetch(`/process_video/${S.videoId}`, { method: 'POST' });
        const data2 = await res2.json();
        if (!res2.ok) { toast('Processing failed: ' + (data2.error || '')); resetVidUI(); return; }

        // Reset stats from the previous video
        S.videoKdrtCount = 0;
        S.videoMaxPct = 0;
        S.videoDet = 'normal';
        S.videoPct = 0;
        S.videoDone = false;

        // Show split-view, hide idle
        S.videoProcessing = true;
        D.vidFeedIdle.classList.add('hidden');
        D.vidSplitView.classList.remove('hidden');
        D.vidBgRaw.src  = `/video_feed/vid_raw/${S.videoId}?t=${Date.now()}`;
        D.vidBgAnno.src = `/video_feed/vid_anno/${S.videoId}?t=${Date.now()}`;

        // Update buttons
        D.btnVidProcess.classList.add('hidden');
        D.btnVidStop.classList.remove('hidden');
        D.vidProgressWrap.style.display = 'flex';

        // Immediately refresh status so it doesn't get stuck on "Waiting"
        refreshVideo();

    } catch (err) {
        toast('Error: ' + err.message);
        resetVidUI();
    }
};

/* -------- Stop -------- */
function stopVideo() {
    if (S.videoId && S.socket && S.connected) {
        S.socket.emit('stop_video', { video_id: S.videoId });
    }
    resetVidUI();
}

D.btnVidStop.onclick = () => stopVideo();

function resetVidUI() {
    S.videoProcessing = false;
    S.videoDone = false;
    S.videoKdrtCount = 0;
    S.videoMaxPct = 0;
    S.videoDet = 'normal';
    S.videoPct = 0;
    D.btnVidProcess.disabled = !S.videoFile;
    D.btnVidProcess.innerHTML = '<svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="2"><polygon points="6,4 18,11 6,18" fill="currentColor" stroke="none"/></svg> Process Video';
    D.btnVidProcess.classList.remove('hidden');
    D.btnVidStop.classList.add('hidden');
    D.vidPbFill.style.width = '0%';
    D.vidPbLabel.textContent = '0%';
}

/* -------- Refresh Video Status Card -------- */
function refreshVideo() {
    const k = S.videoDet === 'kdrt';
    // Status text — real-time during processing
    let statusTxt, statusCls;
    if (S.videoProcessing) {
        // Real-time: status changes according to the current video moment
        if (k) { statusTxt = 'KDRT Detected'; statusCls = 'red'; }
        else   { statusTxt = 'Normal'; statusCls = 'green'; }
    } else if (S.videoDone) {
        // Video finished: show summary
        if (S.videoKdrtCount > 0) { statusTxt = 'KDRT Found'; statusCls = 'red'; }
        else { statusTxt = 'Safe'; statusCls = 'green'; }
    } else {
        statusTxt = 'Waiting'; statusCls = 'yellow';
    }
    D.vidVStatus.textContent = statusTxt;
    D.vidVStatus.className   = 'cs-val ' + statusCls;
    D.vidVPct.textContent    = S.videoPct + '%';
    D.vidVPct.className      = 'cs-val ' + (S.videoPct > 70 ? 'red' : 'green');
    D.vidVPersons.textContent = S.videoPersons;
    D.vidVPersons.className   = 'cs-val ' + (S.videoPersons >= 2 ? 'green' : 'yellow');
    // Description text
    let desc;
    if (S.videoProcessing) {
        // Real-time description while streaming
        if (k) {
            desc = `"Currently detected as <strong class=\"red\">KDRT</strong> — probability: <strong>${S.videoPct}%</strong>"`;
        } else {
            desc = `"Current condition is <strong class=\"green\">Normal</strong> — probability: <strong>${S.videoPct}%</strong>"`;
        }
    } else if (S.videoDone) {
        if (S.videoKdrtCount > 0) {
            desc = `"Video analysis complete — KDRT detected <strong class=\"red\">${S.videoKdrtCount} times</strong> during the video<br>Highest percentage: <strong>${S.videoMaxPct}%</strong>"`;
        } else {
            desc = '"Video analysis complete — <strong class=\"green\">No KDRT detected</strong>"';
        }
    } else {
        desc = '"Upload a video to start KDRT detection analysis"';
    }
    D.vidVDesc.innerHTML = desc;
}

/* -------- WebSocket Video Events -------- */
function initVideoSocketEvents() {
    if (!S.socket) return;

    S.socket.on('video_progress', data => {
        if (data.video_id !== S.videoId) return;
        const pct = data.percent || 0;
        D.vidPbFill.style.width = pct + '%';
        D.vidPbLabel.textContent = pct + '%  (' + data.frame + '/' + data.total + ' frame)';
    });

    S.socket.on('video_detection_update', data => {
        if (data.video_id !== S.videoId) return;
        S.videoPct     = data.percentage || 0;
        S.videoDet     = data.is_alert ? 'kdrt' : 'normal';
        S.videoPersons = data.persons || 0;
        // Track KDRT occurrences and max percentage during the video
        if (data.is_alert) {
            S.videoKdrtCount = (S.videoKdrtCount || 0) + 1;
            S.videoMaxPct = Math.max(S.videoMaxPct || 0, data.percentage || 0);
        }
        if (S.view === 'video') refreshVideo();
    });

    S.socket.on('video_done', data => {
        if (data.video_id !== S.videoId) return;
        toast('Video processing complete');
        S.videoProcessing = false;
        S.videoDone = true;
        refreshVideo();
        D.vidPbFill.style.width  = '100%';
        D.vidPbLabel.textContent = '100% — Done';
        D.btnVidStop.classList.add('hidden');
        D.btnVidProcess.classList.remove('hidden');
        D.btnVidProcess.disabled = false;
        D.btnVidProcess.innerHTML = '<svg viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="2"><polygon points="6,4 18,11 6,18" fill="currentColor" stroke="none"/></svg> Process Again';
    });

    S.socket.on('video_stopped', data => {
        if (data.video_id !== S.videoId) return;
        S.videoProcessing = false;
    });

    S.socket.on('video_error', data => {
        if (data.video_id !== S.videoId) return;
        toast('Error video: ' + data.error);
        resetVidUI();
    });
}

// Attach video WebSocket events — called after socket is ready
// We hook into the existing S.socket.on('connect') inside initSocket by
// using a MutationObserver-free approach: poll until socket is open.
(function waitForSocket() {
    if (S.socket) { initVideoSocketEvents(); }
    else { setTimeout(waitForSocket, 200); }
})();

refreshVideo();


