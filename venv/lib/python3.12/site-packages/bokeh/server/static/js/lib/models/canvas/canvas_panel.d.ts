import { StyledElement, StyledElementView } from "../ui/styled_element";
import type { PlotView } from "../plots/plot";
import type { Node } from "../coordinates/node";
import type * as p from "../../core/properties";
import { InlineStyleSheet } from "../../core/dom";
import type { StyleSheetLike } from "../../core/dom";
import type { XY } from "../../core/util/bbox";
import { BBox } from "../../core/util/bbox";
import { Place } from "../../core/enums";
export declare class CanvasPanelView extends StyledElementView {
    model: CanvasPanel;
    parent: PlotView;
    private _bbox;
    get bbox(): BBox;
    readonly position: InlineStyleSheet;
    stylesheets(): StyleSheetLike[];
    rendering_target(): HTMLElement;
    render(): void;
    set_geometry(bbox: BBox): void;
    /**
     * Updates the position of the associated DOM element.
     */
    protected _update_position(): void;
    resolve_symbol(node: Node): XY | number;
}
export declare namespace CanvasPanel {
    type Attrs = p.AttrsOf<Props>;
    type Props = StyledElement.Props & {
        place: p.Property<Place>;
    };
}
export interface CanvasPanel extends CanvasPanel.Attrs {
}
export declare class CanvasPanel extends StyledElement {
    properties: CanvasPanel.Props;
    __view_type__: CanvasPanelView;
    constructor(attrs?: Partial<CanvasPanel.Attrs>);
}
//# sourceMappingURL=canvas_panel.d.ts.map