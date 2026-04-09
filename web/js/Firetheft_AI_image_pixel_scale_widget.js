import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Firetheft.ImagePixelScaleWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImagePixelScaleNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this;

                // Allow widgets to initialize properly
                setTimeout(() => {
                    const scaleModeWidget = node.widgets.find(w => w.name === "scale_mode");
                    const scaleFactorWidget = node.widgets.find(w => w.name === "scale_factor");
                    const resolutionWidget = node.widgets.find(w => w.name === "resolution");

                    if (!scaleModeWidget || !scaleFactorWidget || !resolutionWidget) return;

                    // Save original properties more robustly
                    scaleFactorWidget.origType = scaleFactorWidget.type;
                    scaleFactorWidget.origComputeSize = scaleFactorWidget.computeSize;

                    // Ensure we capture "combo" if it's a list
                    const isCombo = Array.isArray(nodeData.input?.required?.resolution?.[0]);
                    resolutionWidget.origType = isCombo ? "combo" : resolutionWidget.type;
                    resolutionWidget.origComputeSize = resolutionWidget.computeSize;

                    node.updateVisibility = () => {
                        const mode = scaleModeWidget.value;
                        // Match Python defined strings: "multiple" and "resolution"
                        if (mode === "multiple") {
                            scaleFactorWidget.type = scaleFactorWidget.origType;
                            scaleFactorWidget.computeSize = scaleFactorWidget.origComputeSize;
                            resolutionWidget.type = "hidden";
                            resolutionWidget.computeSize = () => [0, -4];
                        } else if (mode === "resolution") {
                            resolutionWidget.type = resolutionWidget.origType;
                            resolutionWidget.computeSize = resolutionWidget.origComputeSize;
                            scaleFactorWidget.type = "hidden";
                            scaleFactorWidget.computeSize = () => [0, -4];
                        }

                        // Trigger resize and refresh
                        setTimeout(() => {
                            if (node.setSize) {
                                const size = node.computeSize();
                                if (size) {
                                    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
                                }
                                app.graph.setDirtyCanvas(true, true);
                            }
                        }, 20);
                    };

                    node.updateVisibility();

                    const origCallback = scaleModeWidget.callback;
                    scaleModeWidget.callback = function (value, cbApp) {
                        node.updateVisibility();
                        if (origCallback) {
                            return origCallback.apply(this, arguments);
                        }
                    };
                }, 50);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, arguments);
                if (this.updateVisibility) {
                    this.updateVisibility();
                } else {
                    setTimeout(() => {
                        if (this.updateVisibility) this.updateVisibility();
                    }, 100);
                }
            };
        }
    }
});
