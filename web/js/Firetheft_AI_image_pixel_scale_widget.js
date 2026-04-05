import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Firetheft.ImagePixelScaleWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImagePixelScaleNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                
                const node = this;
                
                // Allow widgets to initialize
                setTimeout(() => {
                    const scaleModeWidget = node.widgets.find(w => w.name === "scale_mode");
                    const scaleFactorWidget = node.widgets.find(w => w.name === "scale_factor");
                    const resolutionWidget = node.widgets.find(w => w.name === "resolution");

                    if (!scaleModeWidget || !scaleFactorWidget || !resolutionWidget) return;

                    // Save original properties
                    scaleFactorWidget.origType = scaleFactorWidget.type;
                    scaleFactorWidget.origComputeSize = scaleFactorWidget.computeSize;
                    
                    resolutionWidget.origType = resolutionWidget.type;
                    resolutionWidget.origComputeSize = resolutionWidget.computeSize;

                    node.updateVisibility = () => {
                        const mode = scaleModeWidget.value;
                        if (mode === "按倍数缩放" || mode === "By Scale Factor") {
                            scaleFactorWidget.type = scaleFactorWidget.origType;
                            scaleFactorWidget.computeSize = scaleFactorWidget.origComputeSize;
                            resolutionWidget.type = "hidden";
                            resolutionWidget.computeSize = () => [0, -4];
                        } else {
                            resolutionWidget.type = resolutionWidget.origType;
                            resolutionWidget.computeSize = resolutionWidget.origComputeSize;
                            scaleFactorWidget.type = "hidden";
                            scaleFactorWidget.computeSize = () => [0, -4];
                        }
                        
                        setTimeout(() => {
                            if (node.computeSize) {
                                node.setSize(node.computeSize());
                                app.graph.setDirtyCanvas(true, true);
                            }
                        }, 10);
                    };

                    node.updateVisibility();

                    const origCallback = scaleModeWidget.callback;
                    scaleModeWidget.callback = function(value, cbApp) {
                        node.updateVisibility();
                        if (origCallback) {
                            return origCallback.apply(this, arguments);
                        }
                    };
                }, 10);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                onConfigure?.apply(this, arguments);
                if (this.updateVisibility) {
                    this.updateVisibility();
                } else {
                    setTimeout(() => {
                        if (this.updateVisibility) this.updateVisibility();
                    }, 50);
                }
            };
        }
    }
});
