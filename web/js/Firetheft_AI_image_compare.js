import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "Comfy.FiretheftAI.ImageCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ImageCompareNode") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

            const node = this;

            // 加入属性以便用户可以在 “Properties” 面板中自定义初始宽高
            node.properties = node.properties || {};
            if (!("initial_width" in node.properties)) node.properties["initial_width"] = 300;
            if (!("initial_height" in node.properties)) node.properties["initial_height"] = 370;

            // 重写 computeSize，防止 ComfyUI 自动把节点缩得很小
            const origComputeSize = node.computeSize;
            node.computeSize = function (out) {
                let size = origComputeSize ? origComputeSize.apply(this, arguments) : [100, 100];
                size[0] = Math.max(size[0], this.properties["initial_width"] || 300);
                size[1] = Math.max(size[1], this.properties["initial_height"] || 370);
                return size;
            };

            node.size = node.computeSize();
            node.resizable = true;

            // 当用户在属性对话框中修改属性时，动态应用新尺寸
            node.onPropertyChanged = function (name, value) {
                if (name === "initial_width") {
                    this.size[0] = Math.max(this.size[0], value);
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                }
                if (name === "initial_height") {
                    this.size[1] = Math.max(this.size[1], value);
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                }
            };

            const uniqueId = Math.random().toString(36).substr(2, 9);
            const widgetId = `Firetheft-compare-${uniqueId}`;

            node.sliderPos = 0.5;
            node.imgUrlA = "";
            node.imgUrlB = "";

            let animationFrameId = null;

            function createOverlay() {
                const existing = document.getElementById(widgetId);
                if (existing) existing.remove();

                const container = document.createElement("div");
                container.id = widgetId;

                Object.assign(container.style, {
                    position: "absolute",
                    top: "0",
                    left: "0",
                    width: "0px",
                    height: "0px",
                    overflow: "hidden",
                    zIndex: "800",
                    backgroundColor: "#151515",
                    border: "1px solid #333",
                    borderRadius: "4px",
                    display: "none",
                    userSelect: "none"
                });

                container.innerHTML = `
                    <div class="Firetheft-img-layer Firetheft-img-b" style="position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;">
                        <img src="" style="width:100%;height:100%;object-fit:contain;display:block;">
                        <span class="Firetheft-label-b" style="position:absolute;top:10px;right:10px;color:rgba(255,255,255,0.8);background:rgba(0,0,0,0.6);padding:2px 6px;border-radius:4px;font-size:12px;pointer-events:none;font-family:sans-serif;font-weight:bold;transition: opacity 0.2s ease;">B</span>
                        <span class="Firetheft-dim-b" style="position:absolute;bottom:10px;right:10px;color:rgba(255,255,255,0.6);background:rgba(0,0,0,0.6);padding:2px 4px;border-radius:3px;font-size:10px;pointer-events:none;font-family:monospace;transition: opacity 0.2s ease;"></span>
                    </div>
                    <div class="Firetheft-img-layer Firetheft-img-a" style="position:absolute;top:0;left:0;width:100%;height:100%;z-index:2;">
                        <img src="" style="width:100%;height:100%;object-fit:contain;display:block;">
                        <span class="Firetheft-label-a" style="position:absolute;top:10px;left:10px;color:rgba(255,255,255,0.8);background:rgba(0,0,0,0.6);padding:2px 6px;border-radius:4px;font-size:12px;pointer-events:none;font-family:sans-serif;font-weight:bold;transition: opacity 0.2s ease;">A</span>
                        <span class="Firetheft-dim-a" style="position:absolute;bottom:10px;left:10px;color:rgba(255,255,255,0.6);background:rgba(0,0,0,0.6);padding:2px 4px;border-radius:3px;font-size:10px;pointer-events:none;font-family:monospace;transition: opacity 0.2s ease;"></span>
                    </div>
                    <div class="Firetheft-slider-handle" style="position:absolute;top:0;bottom:0;width:40px;margin-left:-20px;z-index:3;cursor:col-resize;display:flex;justify-content:center;">
                        <div style="width:1px;height:100%;background:#00E5FF;box-shadow: 0 0 4px rgba(0,0,0,0.5);"></div>
                    </div>
                `;

                document.body.appendChild(container);

                container.addEventListener("mousedown", (e) => {
                    if (e.button === 0) {
                        e.preventDefault();
                        const handleUpdate = (clientX) => {
                            const rect = container.getBoundingClientRect();
                            let pos = (clientX - rect.left) / rect.width;
                            pos = Math.max(0, Math.min(1, pos));
                            node.sliderPos = pos;
                            updateVisuals(container, pos);
                        };
                        handleUpdate(e.clientX);

                        const onMouseMove = (moveEvent) => {
                            moveEvent.preventDefault();
                            handleUpdate(moveEvent.clientX);
                        };
                        const onMouseUp = () => {
                            document.removeEventListener("mousemove", onMouseMove);
                            document.removeEventListener("mouseup", onMouseUp);
                            container.style.pointerEvents = "auto";
                        };
                        document.addEventListener("mousemove", onMouseMove);
                        document.addEventListener("mouseup", onMouseUp);
                    }

                    else if (e.button === 1) {
                        e.preventDefault();

                        const startX = e.clientX;
                        const startY = e.clientY;
                        const startOffsets = [...app.canvas.ds.offset];

                        const onPanMove = (moveEvent) => {
                            const dx = moveEvent.clientX - startX;
                            const dy = moveEvent.clientY - startY;

                            app.canvas.ds.offset[0] = startOffsets[0] + dx / app.canvas.ds.scale;
                            app.canvas.ds.offset[1] = startOffsets[1] + dy / app.canvas.ds.scale;

                            app.canvas.setDirty(true, true);
                        };

                        const onPanUp = () => {
                            document.removeEventListener("mousemove", onPanMove);
                            document.removeEventListener("mouseup", onPanUp);
                        };

                        document.addEventListener("mousemove", onPanMove);
                        document.addEventListener("mouseup", onPanUp);
                    }
                });

                container.addEventListener("wheel", (e) => {
                    const canvasEl = app.canvas.canvas;
                    if (canvasEl) {
                        const newEvent = new WheelEvent("wheel", {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            detail: e.detail,
                            deltaX: e.deltaX,
                            deltaY: e.deltaY,
                            deltaZ: e.deltaZ,
                            deltaMode: e.deltaMode,
                            clientX: e.clientX,
                            clientY: e.clientY,
                            screenX: e.screenX,
                            screenY: e.screenY,
                            ctrlKey: e.ctrlKey,
                            altKey: e.altKey,
                            shiftKey: e.shiftKey,
                            metaKey: e.metaKey
                        });
                        canvasEl.dispatchEvent(newEvent);
                    }
                }, { passive: true });

                return container;
            }

            function updateVisuals(container, pos) {
                const pct = pos * 100;
                const imgA = container.querySelector(".Firetheft-img-a");
                const handle = container.querySelector(".Firetheft-slider-handle");
                const labelA = container.querySelector(".Firetheft-label-a");
                const labelB = container.querySelector(".Firetheft-label-b");
                const dimA = container.querySelector(".Firetheft-dim-a");
                const dimB = container.querySelector(".Firetheft-dim-b");

                imgA.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
                handle.style.left = `${pct}%`;

                const hideA = pos < 0.1 ? "0" : "1";
                const hideB = pos > 0.9 ? "0" : "1";

                if (labelA) labelA.style.opacity = hideA;
                if (dimA) dimA.style.opacity = hideA;
                if (labelB) labelB.style.opacity = hideB;
                if (dimB) dimB.style.opacity = hideB;
            }

            function syncLoop() {
                const el = document.getElementById(widgetId);
                if (el) {
                    updateLayout(el);
                }
                if (node.graph) {
                    animationFrameId = requestAnimationFrame(syncLoop);
                }
            }

            function updateLayout(el) {
                if (node.flags.collapsed || !node.imgUrlA) {
                    el.style.display = "none";
                    return;
                }

                const canvas = app.canvas;
                const scale = canvas.ds.scale;
                const offset = canvas.ds.offset;

                const nodeX = node.pos[0];
                const nodeY = node.pos[1];

                const paddingLeft = 0;
                const paddingBottom = 30;
                const margin = 10;
                const headerHeight = 60;

                const finalX = (nodeX + offset[0]) * scale + ((margin + paddingLeft) * scale);
                const finalY = (nodeY + offset[1]) * scale + (headerHeight * scale);

                const finalW = (node.size[0] - margin * 2 - paddingLeft) * scale;
                const finalH = (node.size[1] - headerHeight - margin - paddingBottom) * scale;

                if (finalX + finalW < 0 || finalY + finalH < 0 ||
                    finalX > window.innerWidth || finalY > window.innerHeight) {
                    el.style.display = "none";
                    return;
                }

                el.style.display = "block";
                el.style.transform = `translate(${finalX}px, ${finalY}px)`;
                el.style.width = `${finalW}px`;
                el.style.height = `${finalH}px`;
            }

            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
                const el = document.getElementById(widgetId);
                if (el) el.remove();
                if (animationFrameId) cancelAnimationFrame(animationFrameId);
            };

            const origOnExecuted = this.onExecuted;
            this.onExecuted = function (output) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);

                if (output && output.Firetheft_images && output.Firetheft_images.length >= 2) {
                    let el = document.getElementById(widgetId);
                    if (!el) {
                        el = createOverlay();
                        if (!animationFrameId) syncLoop();
                    }

                    const getUrl = (img) => api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}`);
                    const rand = "&t=" + Date.now();

                    node.imgUrlA = getUrl(output.Firetheft_images[0]) + rand;
                    node.imgUrlB = getUrl(output.Firetheft_images[1]) + rand;

                    const imgElA = el.querySelector(".Firetheft-img-a img");
                    const imgElB = el.querySelector(".Firetheft-img-b img");
                    const dimA = el.querySelector(".Firetheft-dim-a");
                    const dimB = el.querySelector(".Firetheft-dim-b");

                    imgElA.onload = () => {
                        if (dimA) dimA.textContent = `${imgElA.naturalWidth}x${imgElA.naturalHeight}`;
                    }
                    imgElB.onload = () => {
                        if (dimB) dimB.textContent = `${imgElB.naturalWidth}x${imgElB.naturalHeight}`;
                    }

                    imgElA.src = node.imgUrlA;
                    imgElB.src = node.imgUrlB;

                    updateVisuals(el, node.sliderPos);
                }
            };

            setTimeout(() => {
                if (!animationFrameId && node.graph) {
                    syncLoop();
                }
            }, 100);
        };
    },
});