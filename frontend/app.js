/**
 * BrainTumor AI – Frontend Application Logic
 * Handles: volume loading, analysis requests, results display, slice navigation
 */

const API_BASE = 'http://localhost:5000/api';

// ─── State ───
let state = {
    currentScreen: 'home',
    volumes: [],
    selectedVolume: null,
    selectedSlice: null,
    currentView: 'mri',
    analysisResult: null,
    images: {},
};

// ─── DOM Elements ───
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const screens = {
    home: $('#homeScreen'),
    processing: $('#processingScreen'),
    results: $('#resultsScreen'),
};

// ─── Screen Management ───
function showScreen(name) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[name].classList.add('active');
    state.currentScreen = name;
}

// ─── Server Status ───
async function checkServerStatus() {
    const statusEl = $('#serverStatus');
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        
        if (data.status === 'online') {
            statusEl.innerHTML = `
                <div class="status-dot ${data.model_loaded ? 'online' : 'offline'}"></div>
                <span>${data.model_loaded ? 'Model Ready' : 'Model Not Loaded'} · ${data.device.toUpperCase()}</span>
            `;
            return data;
        }
    } catch (e) {
        statusEl.innerHTML = `
            <div class="status-dot offline"></div>
            <span>Server Offline</span>
        `;
        return null;
    }
}

// ─── Load Volumes ───
async function loadVolumes() {
    const grid = $('#volumeGrid');
    try {
        const res = await fetch(`${API_BASE}/volumes`);
        const data = await res.json();
        state.volumes = data.volumes;
        
        grid.innerHTML = '';
        data.volumes.forEach(vol => {
            const card = document.createElement('div');
            card.className = 'volume-card';
            card.innerHTML = `
                <div class="vol-id">Vol ${vol.id}</div>
                <div class="vol-slices">${vol.num_slices} slices</div>
            `;
            card.addEventListener('click', () => selectVolume(vol.id));
            grid.appendChild(card);
        });
    } catch (e) {
        grid.innerHTML = `
            <div class="loading-volumes">
                <span style="color: var(--error)">⚠ Cannot connect to server. Start the backend first.</span>
            </div>
        `;
    }
}

// ─── Select Volume & Start Analysis ───
async function selectVolume(volumeId) {
    state.selectedVolume = volumeId;
    
    // Show processing screen
    showScreen('processing');
    animateProgress();
    
    try {
        // First find best tumor slice
        updateStep(1, 'active');
        setProgress(15, 'Finding tumor slices...');
        
        const tumorRes = await fetch(`${API_BASE}/volume/${volumeId}/tumor-slices?top=1`);
        const tumorData = await tumorRes.json();
        
        let sliceIndex = 77; // default middle
        if (tumorData.tumor_slices && tumorData.tumor_slices.length > 0) {
            sliceIndex = tumorData.tumor_slices[0].slice_index;
        }
        state.selectedSlice = sliceIndex;
        
        // Run analysis
        updateStep(1, 'done');
        updateStep(2, 'active');
        setProgress(35, 'Preprocessing MRI data...');
        
        await delay(400);
        updateStep(2, 'done');
        updateStep(3, 'active');
        setProgress(60, 'Running segmentation model...');
        
        const analyzeRes = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                volume_id: volumeId, 
                slice_index: sliceIndex 
            }),
        });
        
        const result = await analyzeRes.json();
        
        if (result.error) {
            alert(`Error: ${result.error}`);
            showScreen('home');
            return;
        }
        
        updateStep(3, 'done');
        updateStep(4, 'active');
        setProgress(85, 'Generating analysis...');
        
        await delay(400);
        updateStep(4, 'done');
        setProgress(100, 'Complete!');
        
        await delay(500);
        
        // Show results
        state.analysisResult = result;
        displayResults(result);
        showScreen('results');
        
    } catch (e) {
        console.error('Analysis failed:', e);
        alert('Analysis failed. Make sure the backend is running and model is trained.');
        showScreen('home');
    }
}

// ─── Custom MRI Upload ───
async function uploadCustomMRI(file) {
    const validExts = ['.nii', '.nii.gz', '.jpg', '.jpeg', '.png'];
    const fileName = file.name.toLowerCase();
    
    if (!validExts.some(ext => fileName.endsWith(ext))) {
        alert('Invalid file format. Please upload .nii, .nii.gz, .jpg, or .png');
        return;
    }
    
    showScreen('processing');
    animateProgress();
    updateStep(1, 'active');
    setProgress(15, 'Uploading file...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const uploadRes = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        updateStep(1, 'done');
        updateStep(2, 'active');
        setProgress(40, 'Processing volume & extracting slice...');
        
        const result = await uploadRes.json();
        if (result.error) {
            alert(`Error: ${result.error}`);
            showScreen('home');
            return;
        }
        
        updateStep(2, 'done');
        updateStep(3, 'active');
        setProgress(75, 'Running model & analyzing...');
        await delay(500);
        
        updateStep(3, 'done');
        updateStep(4, 'done');
        setProgress(100, 'Complete!');
        await delay(500);
        
        state.analysisResult = result;
        state.selectedVolume = 'Custom Upload';
        state.selectedSlice = result.slice_index;
        displayResults(result);
        showScreen('results');
        
    } catch (e) {
        console.error('Upload failed:', e);
        alert('Upload and analysis failed. Ensure server is running.');
        showScreen('home');
    }
}

// ─── Analyze Specific Slice ───
async function analyzeSlice(volumeId, sliceIndex) {
    showScreen('processing');
    animateProgress();
    
    try {
        updateStep(1, 'active');
        setProgress(20, 'Loading slice...');
        
        await delay(200);
        updateStep(1, 'done');
        updateStep(2, 'active');
        setProgress(40, 'Preprocessing...');
        
        await delay(200);
        updateStep(2, 'done');
        updateStep(3, 'active');
        setProgress(60, 'Running model...');
        
        const res = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ volume_id: volumeId, slice_index: sliceIndex }),
        });
        
        const result = await res.json();
        
        if (result.error) {
            alert(`Error: ${result.error}`);
            showScreen('results');
            return;
        }
        
        updateStep(3, 'done');
        updateStep(4, 'active');
        setProgress(90, 'Finalizing...');
        
        await delay(300);
        updateStep(4, 'done');
        setProgress(100, 'Done!');
        
        await delay(300);
        
        state.analysisResult = result;
        state.selectedSlice = sliceIndex;
        displayResults(result);
        showScreen('results');
        
    } catch (e) {
        console.error('Slice analysis failed:', e);
        alert('Analysis failed.');
        showScreen('results');
    }
}

// ─── Display Results ───
function displayResults(result) {
    // Meta
    $('#resultMeta').textContent = `Volume ${result.volume_id} · Slice ${result.slice_index}`;
    
    // Check if no tumor
    if (result.tumor_detected === false) {
        // Hide segmentation elements, show no tumor message
        $('#tumorBadge').className = 'result-badge no-tumor';
        $('#tumorBadge').innerHTML = '<span class="badge-dot"></span> No Tumor Detected';
        
        $('#predDetected').textContent = '❌ No';
        $('#predType').textContent = 'None';
        $('#predLocation').textContent = 'N/A';
        $('#predConfidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;
        $('#predArea').textContent = '0%';
        
        // Hide dice and composition
        $$('.info-card').forEach(card => {
            if (card.classList.contains('card-metrics') || card.classList.contains('card-composition')) {
                card.style.display = 'none';
            }
        });
        
        // Show only MRI
        setViewTab('mri');
        
        // Insights
        renderInsights([{
            type: 'info',
            title: 'No Tumor Found',
            text: result.message || 'The classifier detected no tumor in this image.'
        }]);
        
        return;
    }
    
    // Normal tumor case
    // Store images
    state.images = result.images;
    
    // Show default view
    setViewTab('pred_overlay');
    
    // Tumor badge
    const badge = $('#tumorBadge');
    if (result.tumor_info.detected) {
        badge.className = 'result-badge';
        badge.innerHTML = '<span class="badge-dot"></span> Tumor Detected';
    } else {
        badge.className = 'result-badge no-tumor';
        badge.innerHTML = '<span class="badge-dot"></span> No Tumor Found';
    }
    
    // Prediction panel
    $('#predDetected').textContent = result.tumor_info.detected ? '✔ Yes' : '✘ No';
    $('#predType').textContent = result.tumor_info.type;
    $('#predLocation').textContent = result.tumor_info.location;
    $('#predConfidence').textContent = `${(result.metrics.confidence * 100).toFixed(1)}%`;
    $('#predArea').textContent = `${result.tumor_info.stats.tumor_percentage}%`;
    
    // Dice scores
    const dice = result.metrics.dice_scores;
    $('#diceOverall').textContent = dice.overall.toFixed(4);
    $('#diceWT').textContent = dice.wt.toFixed(4);
    $('#diceTC').textContent = dice.tc.toFixed(4);
    $('#diceET').textContent = dice.et.toFixed(4);
    
    // Animate bars
    setTimeout(() => {
        $('#barOverall').style.width = `${dice.overall * 100}%`;
        $('#barWT').style.width = `${dice.wt * 100}%`;
        $('#barTC').style.width = `${dice.tc * 100}%`;
        $('#barET').style.width = `${dice.et * 100}%`;
    }, 100);
    
    // Composition
    const stats = result.tumor_info.stats;
    $('#totalTumor').textContent = `${stats.tumor_percentage}%`;
    $('#compWT').textContent = `${stats.wt_percent}%`;
    $('#compTC').textContent = `${stats.tc_percent}%`;
    $('#compET').textContent = `${stats.et_percent}%`;
    
    drawCompositionRing(stats.wt_percent, stats.tc_percent, stats.et_percent);
    
    // Insights
    renderInsights(result.insights);
    
    // Slice slider
    $('#sliceSlider').value = result.slice_index;
    $('#sliceLabel').textContent = `Slice ${result.slice_index}`;
}

// ─── Image Viewer Tabs ───
function setViewTab(viewName) {
    state.currentView = viewName;
    
    $$('.view-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === viewName);
    });
    
    const img = $('#viewerImage');
    if (state.images[viewName]) {
        img.src = `data:image/png;base64,${state.images[viewName]}`;
        img.alt = `MRI ${viewName}`;
    }
}

// ─── Composition Donut Chart ───
function drawCompositionRing(wt, tc, et) {
    const canvas = $('#compositionCanvas');
    const ctx = canvas.getContext('2d');
    const cx = 80, cy = 80, radius = 60, lineWidth = 14;
    
    ctx.clearRect(0, 0, 160, 160);
    
    const total = wt + tc + et;
    if (total === 0) {
        // Empty ring
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = lineWidth;
        ctx.stroke();
        return;
    }
    
    const segments = [
        { pct: wt / total, color: '#FFD700' },
        { pct: tc / total, color: '#FF8C00' },
        { pct: et / total, color: '#FF0000' },
    ];
    
    let startAngle = -Math.PI / 2;
    
    segments.forEach(seg => {
        if (seg.pct <= 0) return;
        const sweep = seg.pct * Math.PI * 2;
        
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, startAngle + sweep);
        ctx.strokeStyle = seg.color;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();
        
        startAngle += sweep;
    });
}

// ─── Insights Renderer ───
function renderInsights(insights) {
    const container = $('#insightsContainer');
    container.innerHTML = '';
    
    if (!insights || insights.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No insights available.</p>';
        return;
    }
    
    insights.forEach(insight => {
        const div = document.createElement('div');
        div.className = `insight-item ${insight.type}`;
        div.innerHTML = `
            <div class="insight-title">${insight.title}</div>
            <div class="insight-text">${insight.text}</div>
        `;
        container.appendChild(div);
    });
}

// ─── Progress Animation ───
function animateProgress() {
    setProgress(0, 'Initializing...');
    // Reset steps
    for (let i = 1; i <= 4; i++) {
        const step = $(`#step${i}`);
        step.className = 'step';
    }
}

function setProgress(pct, text) {
    $('#progressBar').style.width = `${pct}%`;
    if (text) $('#processingText').textContent = text;
}

function updateStep(num, status) {
    const step = $(`#step${num}`);
    step.className = `step ${status}`;
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Event Listeners ───
document.addEventListener('DOMContentLoaded', () => {
    // Check server & load volumes
    checkServerStatus();
    loadVolumes();
    
    // Upload event listeners
    const dropzone = $('#uploadDropzone');
    const fileInput = $('#fileInput');
    const browseText = $('#browseFileText');
    
    browseText.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadCustomMRI(e.target.files[0]);
        }
    });
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            uploadCustomMRI(e.dataTransfer.files[0]);
        }
    });

    // Periodic status check
    setInterval(checkServerStatus, 10000);
    
    // View tabs
    $$('.view-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            setViewTab(tab.dataset.view);
        });
    });
    
    // Back button
    $('#btnBack').addEventListener('click', () => {
        showScreen('home');
    });
    
    // Slice navigation
    $('#sliceSlider').addEventListener('input', (e) => {
        $('#sliceLabel').textContent = `Slice ${e.target.value}`;
    });
    
    $('#btnPrevSlice').addEventListener('click', () => {
        const slider = $('#sliceSlider');
        slider.value = Math.max(0, parseInt(slider.value) - 1);
        $('#sliceLabel').textContent = `Slice ${slider.value}`;
    });
    
    $('#btnNextSlice').addEventListener('click', () => {
        const slider = $('#sliceSlider');
        slider.value = Math.min(154, parseInt(slider.value) + 1);
        $('#sliceLabel').textContent = `Slice ${slider.value}`;
    });
    
    $('#btnAnalyzeSlice').addEventListener('click', () => {
        const sliceIdx = parseInt($('#sliceSlider').value);
        if (state.selectedVolume) {
            analyzeSlice(state.selectedVolume, sliceIdx);
        }
    });
});
