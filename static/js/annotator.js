/**
 * EasiVisi - Image Annotator
 * Canvas-based bounding box annotation tool for YOLO training
 */

class ImageAnnotator {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.container = this.canvas.parentElement;

        // Options
        this.classes = options.classes || [];
        this.onSave = options.onSave || (() => { });
        this.onChange = options.onChange || (() => { });

        // State
        this.image = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.annotations = [];
        this.selectedIndex = -1;
        this.currentClassId = 0;
        this.tool = 'box'; // 'box' or 'select'
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        this.hasUnsavedChanges = false;

        // Drawing state
        this.isDrawing = false;
        this.isDragging = false;
        this.isResizing = false;
        this.startPoint = { x: 0, y: 0 };
        this.currentBox = null;
        this.resizeHandle = null;

        // History for undo/redo
        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 50;

        // Initialize
        this.setupCanvas();
        this.setupEventListeners();
    }

    setupCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.render();
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseUp(e));

        // Wheel for zoom
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => this.handleTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.handleTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this.handleTouchEnd(e));
    }

    // Coordinate conversion
    screenToImage(x, y) {
        const offset = this.getImageOffset();
        const scale = this.getScale();
        return {
            x: (x - offset.x) / scale,
            y: (y - offset.y) / scale
        };
    }

    imageToScreen(x, y) {
        const offset = this.getImageOffset();
        const scale = this.getScale();
        return {
            x: x * scale + offset.x,
            y: y * scale + offset.y
        };
    }

    getScale() {
        if (!this.image) return 1;
        const scaleX = this.canvas.width / this.imageWidth;
        const scaleY = this.canvas.height / this.imageHeight;
        return Math.min(scaleX, scaleY) * this.zoom;
    }

    getImageOffset() {
        const scale = this.getScale();
        const scaledWidth = this.imageWidth * scale;
        const scaledHeight = this.imageHeight * scale;
        return {
            x: (this.canvas.width - scaledWidth) / 2 + this.pan.x,
            y: (this.canvas.height - scaledHeight) / 2 + this.pan.y
        };
    }

    // Mouse handlers
    handleMouseDown(e) {
        if (!this.image) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const imgPoint = this.screenToImage(x, y);

        // Check if clicking on resize handle
        if (this.selectedIndex !== -1 && this.tool === 'select') {
            const handle = this.getResizeHandle(x, y);
            if (handle) {
                this.isResizing = true;
                this.resizeHandle = handle;
                this.startPoint = { x: imgPoint.x, y: imgPoint.y };
                return;
            }
        }

        // Check if clicking on existing box (select mode)
        if (this.tool === 'select') {
            const clickedIndex = this.findBoxAtPoint(imgPoint.x, imgPoint.y);
            if (clickedIndex !== -1) {
                this.selectedIndex = clickedIndex;
                this.isDragging = true;
                this.startPoint = { x: imgPoint.x, y: imgPoint.y };
                this.render();
                return;
            } else {
                this.selectedIndex = -1;
                this.render();
            }
        }

        // Start drawing new box
        if (this.tool === 'box') {
            this.isDrawing = true;
            this.startPoint = { x: imgPoint.x, y: imgPoint.y };
            this.currentBox = null;
        }
    }

    handleMouseMove(e) {
        if (!this.image) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const imgPoint = this.screenToImage(x, y);

        // Update cursor based on context
        this.updateCursor(x, y);

        // Handle resizing
        if (this.isResizing && this.selectedIndex !== -1) {
            this.resizeBox(imgPoint.x, imgPoint.y);
            this.render();
            return;
        }

        // Handle dragging
        if (this.isDragging && this.selectedIndex !== -1) {
            const dx = imgPoint.x - this.startPoint.x;
            const dy = imgPoint.y - this.startPoint.y;
            const box = this.annotations[this.selectedIndex];

            box.x = Math.max(0, Math.min(this.imageWidth - box.width, box.x + dx));
            box.y = Math.max(0, Math.min(this.imageHeight - box.height, box.y + dy));

            this.startPoint = { x: imgPoint.x, y: imgPoint.y };
            this.render();
            return;
        }

        // Handle drawing
        if (this.isDrawing) {
            const x1 = Math.min(this.startPoint.x, imgPoint.x);
            const y1 = Math.min(this.startPoint.y, imgPoint.y);
            const x2 = Math.max(this.startPoint.x, imgPoint.x);
            const y2 = Math.max(this.startPoint.y, imgPoint.y);

            this.currentBox = {
                x: Math.max(0, x1),
                y: Math.max(0, y1),
                width: Math.min(this.imageWidth, x2) - Math.max(0, x1),
                height: Math.min(this.imageHeight, y2) - Math.max(0, y1),
                class_id: this.currentClassId
            };

            this.render();
        }
    }

    handleMouseUp(e) {
        if (this.isResizing) {
            this.isResizing = false;
            this.resizeHandle = null;
            this.saveState();
            this.hasUnsavedChanges = true;
            this.onChange();
        }

        if (this.isDragging) {
            this.isDragging = false;
            this.saveState();
            this.hasUnsavedChanges = true;
            this.onChange();
        }

        if (this.isDrawing && this.currentBox) {
            // Only add if box has reasonable size
            if (this.currentBox.width > 5 && this.currentBox.height > 5) {
                this.annotations.push({ ...this.currentBox });
                this.selectedIndex = this.annotations.length - 1;
                this.saveState();
                this.hasUnsavedChanges = true;
                this.onChange();
            }
            this.currentBox = null;
        }

        this.isDrawing = false;
        this.render();
    }

    handleWheel(e) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        this.zoom = Math.max(0.1, Math.min(5, this.zoom * delta));
        document.getElementById('zoomLevel').textContent = `${Math.round(this.zoom * 100)}%`;
        this.render();
    }

    // Touch handlers
    handleTouchStart(e) {
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            this.handleMouseDown({ clientX: touch.clientX, clientY: touch.clientY });
        }
    }

    handleTouchMove(e) {
        e.preventDefault();
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            this.handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
        }
    }

    handleTouchEnd(e) {
        this.handleMouseUp({});
    }

    // Box manipulation
    findBoxAtPoint(x, y) {
        for (let i = this.annotations.length - 1; i >= 0; i--) {
            const box = this.annotations[i];
            if (x >= box.x && x <= box.x + box.width &&
                y >= box.y && y <= box.y + box.height) {
                return i;
            }
        }
        return -1;
    }

    getResizeHandle(screenX, screenY) {
        if (this.selectedIndex === -1) return null;

        const box = this.annotations[this.selectedIndex];
        const corners = [
            { name: 'nw', x: box.x, y: box.y },
            { name: 'ne', x: box.x + box.width, y: box.y },
            { name: 'sw', x: box.x, y: box.y + box.height },
            { name: 'se', x: box.x + box.width, y: box.y + box.height }
        ];

        const handleSize = 10 / this.getScale();

        for (const corner of corners) {
            const screenCorner = this.imageToScreen(corner.x, corner.y);
            if (Math.abs(screenX - screenCorner.x) < handleSize &&
                Math.abs(screenY - screenCorner.y) < handleSize) {
                return corner.name;
            }
        }

        return null;
    }

    resizeBox(imgX, imgY) {
        const box = this.annotations[this.selectedIndex];

        switch (this.resizeHandle) {
            case 'nw':
                box.width += box.x - imgX;
                box.height += box.y - imgY;
                box.x = imgX;
                box.y = imgY;
                break;
            case 'ne':
                box.width = imgX - box.x;
                box.height += box.y - imgY;
                box.y = imgY;
                break;
            case 'sw':
                box.width += box.x - imgX;
                box.x = imgX;
                box.height = imgY - box.y;
                break;
            case 'se':
                box.width = imgX - box.x;
                box.height = imgY - box.y;
                break;
        }

        // Ensure positive dimensions
        if (box.width < 0) {
            box.x += box.width;
            box.width = Math.abs(box.width);
        }
        if (box.height < 0) {
            box.y += box.height;
            box.height = Math.abs(box.height);
        }

        // Clamp to image bounds
        box.x = Math.max(0, box.x);
        box.y = Math.max(0, box.y);
        box.width = Math.min(this.imageWidth - box.x, box.width);
        box.height = Math.min(this.imageHeight - box.y, box.height);
    }

    updateCursor(screenX, screenY) {
        if (this.tool === 'box') {
            this.canvas.style.cursor = 'crosshair';
            return;
        }

        if (this.selectedIndex !== -1) {
            const handle = this.getResizeHandle(screenX, screenY);
            if (handle) {
                const cursors = {
                    'nw': 'nwse-resize',
                    'se': 'nwse-resize',
                    'ne': 'nesw-resize',
                    'sw': 'nesw-resize'
                };
                this.canvas.style.cursor = cursors[handle];
                return;
            }
        }

        const imgPoint = this.screenToImage(screenX, screenY);
        if (this.findBoxAtPoint(imgPoint.x, imgPoint.y) !== -1) {
            this.canvas.style.cursor = 'move';
        } else {
            this.canvas.style.cursor = 'default';
        }
    }

    // Rendering
    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.image) return;

        const offset = this.getImageOffset();
        const scale = this.getScale();

        // Draw image
        this.ctx.drawImage(
            this.image,
            offset.x, offset.y,
            this.imageWidth * scale,
            this.imageHeight * scale
        );

        // Draw existing annotations
        this.annotations.forEach((box, index) => {
            this.drawBox(box, index === this.selectedIndex);
        });

        // Draw current drawing box
        if (this.currentBox) {
            this.drawBox(this.currentBox, false, true);
        }
    }

    drawBox(box, isSelected, isDrawing = false) {
        const scale = this.getScale();
        const offset = this.getImageOffset();

        const x = box.x * scale + offset.x;
        const y = box.y * scale + offset.y;
        const width = box.width * scale;
        const height = box.height * scale;

        // Get class color
        const classInfo = this.classes.find(c => c.id === box.class_id) || { color: '#00ff00' };
        const color = classInfo.color;

        // Draw fill - only show fill when actively drawing for visibility
        if (isDrawing) {
            this.ctx.fillStyle = `${color}40`;
            this.ctx.fillRect(x, y, width, height);
        }

        // Draw border
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = isSelected ? 3 : 2;
        this.ctx.strokeRect(x, y, width, height);

        // Draw label
        if (!isDrawing && classInfo.name) {
            this.ctx.font = '12px Inter, sans-serif';
            const labelWidth = this.ctx.measureText(classInfo.name).width + 10;
            const labelHeight = 20;

            this.ctx.fillStyle = color;
            this.ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);

            this.ctx.fillStyle = '#fff';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(classInfo.name, x + 5, y - labelHeight / 2);
        }

        // Draw resize handles if selected
        if (isSelected) {
            const handleSize = 8;
            const corners = [
                { x: x, y: y },
                { x: x + width, y: y },
                { x: x, y: y + height },
                { x: x + width, y: y + height }
            ];

            this.ctx.fillStyle = '#fff';
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;

            corners.forEach(corner => {
                this.ctx.fillRect(
                    corner.x - handleSize / 2,
                    corner.y - handleSize / 2,
                    handleSize,
                    handleSize
                );
                this.ctx.strokeRect(
                    corner.x - handleSize / 2,
                    corner.y - handleSize / 2,
                    handleSize,
                    handleSize
                );
            });
        }
    }

    // Public API
    async loadImage(url, width, height) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this.imageWidth = width || img.naturalWidth;
                this.imageHeight = height || img.naturalHeight;
                this.annotations = [];
                this.selectedIndex = -1;
                this.zoom = 1;
                this.pan = { x: 0, y: 0 };
                this.history = [];
                this.historyIndex = -1;
                this.hasUnsavedChanges = false;
                document.getElementById('zoomLevel').textContent = '100%';
                this.render();
                resolve();
            };
            img.onerror = reject;
            img.src = url;
        });
    }

    loadAnnotations(annotations) {
        // Backend now sends x,y as top-left corner, width, height in pixels
        this.annotations = annotations.map(ann => ({
            x: ann.x,
            y: ann.y,
            width: ann.width,
            height: ann.height,
            class_id: ann.class_id
        }));
        this.saveState();
        this.render();
        this.onChange();
    }

    getAnnotations() {
        // Return in a format suitable for saving
        return this.annotations.map(box => ({
            class_id: box.class_id,
            x: box.x,
            y: box.y,
            width: box.width,
            height: box.height
        }));
    }

    getImageDimensions() {
        return { width: this.imageWidth, height: this.imageHeight };
    }

    setCurrentClass(classId) {
        this.currentClassId = classId;
    }

    setTool(tool) {
        this.tool = tool;
        document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
        if (tool === 'box') {
            document.getElementById('boxTool').classList.add('active');
        } else {
            document.getElementById('selectTool').classList.add('active');
        }
    }

    deleteAnnotation(index) {
        if (index >= 0 && index < this.annotations.length) {
            this.annotations.splice(index, 1);
            this.selectedIndex = -1;
            this.saveState();
            this.hasUnsavedChanges = true;
            this.render();
            this.onChange();
        }
    }

    deleteSelected() {
        if (this.selectedIndex !== -1) {
            this.deleteAnnotation(this.selectedIndex);
        }
    }

    clearAnnotations() {
        if (this.annotations.length > 0 && confirm('Clear all annotations?')) {
            this.annotations = [];
            this.selectedIndex = -1;
            this.saveState();
            this.hasUnsavedChanges = true;
            this.render();
            this.onChange();
        }
    }

    // Zoom controls
    zoomIn() {
        this.zoom = Math.min(5, this.zoom * 1.2);
        document.getElementById('zoomLevel').textContent = `${Math.round(this.zoom * 100)}%`;
        this.render();
    }

    zoomOut() {
        this.zoom = Math.max(0.1, this.zoom / 1.2);
        document.getElementById('zoomLevel').textContent = `${Math.round(this.zoom * 100)}%`;
        this.render();
    }

    resetZoom() {
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        document.getElementById('zoomLevel').textContent = '100%';
        this.render();
    }

    // Undo/Redo
    saveState() {
        // Remove any states after current index
        this.history = this.history.slice(0, this.historyIndex + 1);

        // Add current state
        this.history.push(JSON.stringify(this.annotations));
        this.historyIndex++;

        // Limit history size
        if (this.history.length > this.maxHistory) {
            this.history.shift();
            this.historyIndex--;
        }
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.annotations = JSON.parse(this.history[this.historyIndex]);
            this.selectedIndex = -1;
            this.hasUnsavedChanges = true;
            this.render();
            this.onChange();
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.annotations = JSON.parse(this.history[this.historyIndex]);
            this.selectedIndex = -1;
            this.hasUnsavedChanges = true;
            this.render();
            this.onChange();
        }
    }

    hasChanges() {
        return this.hasUnsavedChanges;
    }

    markSaved() {
        this.hasUnsavedChanges = false;
    }
}
