const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const processingView = document.getElementById('processing-view');
const imageGrid = document.getElementById('image-grid');
const runBtn = document.getElementById('run-btn');
const resetBtn = document.getElementById('reset-btn');
const iouSlider = document.getElementById('iou-threshold');
const nmsSlider = document.getElementById('nms-threshold');
const iouValue = document.getElementById('iou-threshold-value');
const nmsValue = document.getElementById('nms-threshold-value');
const presenceDetectionToggle = document.getElementById('presence-detection');
const dropSummary = document.getElementById('drop-summary');
const urlApiBase = new URLSearchParams(window.location.search).get('api');
const API_BASE = (urlApiBase || localStorage.getItem('SAM_GC_API_BASE') || 'http://localhost:8000').replace(/\/$/, '');

if (urlApiBase) {
    localStorage.setItem('SAM_GC_API_BASE', API_BASE);
}

let uploadedImages = [];
let originalFiles = [];  // Keep File objects for API

function sliderToFloat(slider) {
    return Number(slider.value) / 100;
}

function syncSliderLabels() {
    iouValue.innerText = sliderToFloat(iouSlider).toFixed(2);
    nmsValue.innerText = sliderToFloat(nmsSlider).toFixed(2);
}

iouSlider.addEventListener('input', syncSliderLabels);
nmsSlider.addEventListener('input', syncSliderLabels);
syncSliderLabels();

function setDropSummary(text) {
    if (dropSummary) {
        dropSummary.innerText = text;
    }
}

function updateDropState(fileCount) {
    const countText = fileCount === 1 ? '1 image selected' : `${fileCount} images selected`;
    const mode = fileCount >= 3 ? 'Ready to execute consensus.' : 'Add at least 3 images.';
    setDropSummary(`${countText} · ${mode}`);
}

// UI Interactions
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

function handleFiles(files) {
    if (!files || files.length < 1) return;

    uploadedImages = [];
    originalFiles = Array.from(files);
    imageGrid.innerHTML = '';
    setDropSummary('Preparing previews...');

    originalFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImages.push({
                id: index,
                src: e.target.result,
                name: file.name
            });
            renderCard(index, e.target.result);
            if (uploadedImages.length === originalFiles.length) {
                updateDropState(originalFiles.length);
            }
        };
        reader.readAsDataURL(file);
    });

    dropZone.classList.add('hidden');
    processingView.classList.remove('hidden');
}

function renderCard(index, src) {
    const card = document.createElement('div');
    card.className = 'image-card relative aspect-square overflow-hidden';
    card.id = `card-${index}`;
    card.innerHTML = `
        <img src="${src}" alt="Uploaded image ${index + 1}">
        <canvas id="mask-canvas-${index}" class="mask-canvas"></canvas>
        <div class="image-label">IMAGE ${index + 1}</div>
        <div id="object-label-${index}" class="image-label" style="top: 1rem; bottom: auto; background: rgba(14, 116, 144, 0.8); border-color: rgba(56, 189, 248, 0.35); opacity: 0; transition: opacity 0.25s ease;"></div>
        <div id="loader-${index}" class="loader-overlay show">
            <div class="loader-spinner"></div>
        </div>
    `;
    imageGrid.appendChild(card);
}

async function runPipeline() {
    if (uploadedImages.length < 3) {
        alert('Please upload at least 3 images for group consensus reasoning.');
        return;
    }

    runBtn.disabled = true;
    runBtn.innerText = 'Contacting SAM-GC...';
    document.getElementById('status-indicator').classList.add('animate-pulse');

    uploadedImages.forEach((_, i) => {
        const loader = document.getElementById(`loader-${i}`);
        if (loader) loader.classList.add('show');
    });

    try {
        updateStep(1);

        const formData = new FormData();
        originalFiles.forEach(file => formData.append('images', file));
        formData.append('iou_threshold', sliderToFloat(iouSlider).toFixed(2));
        formData.append('nms_threshold', sliderToFloat(nmsSlider).toFixed(2));
        formData.append('presence_detection', presenceDetectionToggle.checked ? 'true' : 'false');

        const response = await fetch(`${API_BASE}/api/process-group`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errBody = await response.text();
            throw new Error(`API error: ${response.status} ${errBody}`);
        }

        const result = await response.json();

        updateStep(2);
        updateStep(3);

        document.getElementById('metric-affinity').innerText = Number(result.global_affinity ?? 0).toFixed(3);
        document.getElementById('metric-time').innerText = `${Number(result.graph_solver_time ?? result.process_time ?? 0).toFixed(3)}s`;
        document.getElementById('metric-nodes').innerText = String(result.nodes_evaluated ?? 0);
        document.getElementById('metric-cohesion').innerText = result.group_cohesion ?? 'N/A';
        document.getElementById('common-object-label').innerText = result.common_object_label || 'Uncertain';

        updateStep(4);

        result.winning_masks.forEach((b64mask, i) => {
            const canvas = document.getElementById(`mask-canvas-${i}`);
            const img = new Image();
            img.onload = () => {
                const width = canvas.offsetWidth;
                const height = canvas.offsetHeight;
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, width, height);
                ctx.drawImage(img, 0, 0, width, height);
                canvas.style.opacity = '1';
            };
            img.src = b64mask;

            const loader = document.getElementById(`loader-${i}`);
            if (loader) loader.classList.remove('show');

            const labelChip = document.getElementById(`object-label-${i}`);
            const label = result.similar_object_labels?.[i] || 'uncertain object';
            const conf = result.similar_object_confidences?.[i];
            const confSuffix = Number.isFinite(conf) ? ` (${(conf * 100).toFixed(0)}%)` : '';
            labelChip.innerText = `${label}${confSuffix}`;
            labelChip.style.opacity = '1';
        });

    } catch (error) {
        console.error('Pipeline error:', error);
        document.getElementById('metric-affinity').innerText = 'Error';
        alert(`SAM-GC Error: ${error.message}. Active API: ${API_BASE}`);
    }

    runBtn.disabled = false;
    runBtn.innerText = 'Run SAM-GC Again';
    document.getElementById('status-indicator').classList.replace('bg-blue-500', 'bg-green-500');
}

function updateStep(step) {
    for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`step-${i}`);
        if (el) {
            el.className = 'step-card';
        }
    }
    const active = document.getElementById(`step-${step}`);
    if (active) {
        active.classList.add('step-active');
    }
}

runBtn.addEventListener('click', runPipeline);

resetBtn.addEventListener('click', () => {
    window.location.reload();
});

