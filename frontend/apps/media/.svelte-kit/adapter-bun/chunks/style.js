import "./async.js";
import { q as hasContext, g as getContext, c as setContext, e as attributes, f as clsx, k as stringify, d as derived, m as escape_html, h as attr, i as attr_class, n as attr_style, j as bind_props, l as ensure_array_like, a as spread_props } from "./renderer.js";
import { M as MediaQuery, o as onDestroy } from "./index-server.js";
import { addEdge as addEdge$1, createDevWarn, Position, ConnectionMode, isEdgeBase, isNodeBase, getBezierPath, getSmoothStepPath, getStraightPath, getNodesInside, isEdgeVisible, getEdgePosition, getElevatedEdgeZIndex, adoptUserNodes, updateConnectionLookup, infiniteExtent, SelectionMode, mergeAriaLabelConfig, initialConnection, pointToRendererPoint, createMarkerIds, fitViewport, getInternalNodesBounds, getViewportForBounds, errorMessages, panBy, snapPosition, calculateNodePosition, updateNodeInternals, updateAbsolutePositions, getHandlePosition, nodeHasDimensions, getMarkerId, MarkerType, isNumeric, getNodesBounds, rendererPointToPoint, getElementsToRemove, isRectObject, getOverlappingArea, nodeToRect, evaluateAbsolutePosition, isMacOs, getConnectionStatus, ConnectionLineType, PanOnScrollMode } from "@xyflow/system";
import { f as is_array, j as get_prototype_of, o as object_prototype, t as to_array, k as fallback, r as run, l as exclude_from_object } from "./render-context.js";
import "clsx";
const empty = [];
function snapshot(value, skip_warning = false, no_tojson = false) {
  return clone(value, /* @__PURE__ */ new Map(), "", empty, null, no_tojson);
}
function clone(value, cloned, path, paths, original = null, no_tojson = false) {
  if (typeof value === "object" && value !== null) {
    var unwrapped = cloned.get(value);
    if (unwrapped !== void 0) return unwrapped;
    if (value instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(value)
    );
    if (value instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(value)
    );
    if (is_array(value)) {
      var copy = (
        /** @type {Snapshot<any>} */
        Array(value.length)
      );
      cloned.set(value, copy);
      if (original !== null) {
        cloned.set(original, copy);
      }
      for (var i = 0; i < value.length; i += 1) {
        var element = value[i];
        if (i in value) {
          copy[i] = clone(element, cloned, path, paths, null, no_tojson);
        }
      }
      return copy;
    }
    if (get_prototype_of(value) === object_prototype) {
      copy = {};
      cloned.set(value, copy);
      if (original !== null) {
        cloned.set(original, copy);
      }
      for (var key2 of Object.keys(value)) {
        copy[key2] = clone(
          // @ts-expect-error
          value[key2],
          cloned,
          path,
          paths,
          null,
          no_tojson
        );
      }
      return copy;
    }
    if (value instanceof Date) {
      return (
        /** @type {Snapshot<T>} */
        structuredClone(value)
      );
    }
    if (typeof /** @type {T & { toJSON?: any } } */
    value.toJSON === "function" && !no_tojson) {
      return clone(
        /** @type {T & { toJSON(): any } } */
        value.toJSON(),
        cloned,
        path,
        paths,
        // Associate the instance with the toJSON clone
        value
      );
    }
  }
  if (value instanceof EventTarget) {
    return (
      /** @type {Snapshot<T>} */
      value
    );
  }
  try {
    return (
      /** @type {Snapshot<T>} */
      structuredClone(value)
    );
  } catch (e) {
    return (
      /** @type {Snapshot<T>} */
      value
    );
  }
}
const defaultOnError = createDevWarn("Svelte Flow", "https://svelteflow.dev/");
function addEdge(edgeParams, edges, options = {}) {
  return addEdge$1(edgeParams, edges, {
    ...options,
    onError: options.onError ?? defaultOnError
  });
}
function createContext() {
  const key2 = {};
  return [
    (errorMessage) => {
      if (errorMessage && !hasContext(key2)) {
        throw new Error(errorMessage);
      }
      return getContext(key2);
    },
    (context) => setContext(key2, context)
  ];
}
const [getNodeIdContext, setNodeIdContext] = createContext();
const [getNodeConnectableContext, setNodeConnectableContext] = createContext();
const [getEdgeIdContext, setEdgeIdContext] = createContext();
function Handle($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      id: handleId = null,
      type = "source",
      position = Position.Top,
      style,
      class: className,
      isConnectable: isConnectableProp,
      isConnectableStart = true,
      isConnectableEnd = true,
      isValidConnection,
      onconnect,
      ondisconnect,
      children,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    const nodeId = getNodeIdContext("Handle must be used within a Custom Node component");
    const isConnectableContext = getNodeConnectableContext("Handle must be used within a Custom Node component");
    let isTarget = derived(() => type === "target");
    let isConnectable = derived(() => isConnectableProp !== void 0 ? isConnectableProp : isConnectableContext.value);
    let store = useStore();
    let ariaLabelConfig = derived(() => store.ariaLabelConfig);
    let $$d = derived(() => {
      if (!store.connection.inProgress) {
        return [false, false, false, false, null];
      }
      const { fromHandle, toHandle, isValid } = store.connection;
      const connectingFrom2 = fromHandle && fromHandle.nodeId === nodeId && fromHandle.type === type && fromHandle.id === handleId;
      const connectingTo2 = toHandle && toHandle.nodeId === nodeId && toHandle.type === type && toHandle.id === handleId;
      const isPossibleTargetHandle2 = store.connectionMode === ConnectionMode.Strict ? fromHandle?.type !== type : nodeId !== fromHandle?.nodeId || handleId !== fromHandle?.id;
      const valid2 = connectingTo2 && isValid;
      return [
        true,
        connectingFrom2,
        connectingTo2,
        isPossibleTargetHandle2,
        valid2
      ];
    }), $$derived_array = derived(() => to_array($$d(), 5)), connectionInProgress = derived(() => $$derived_array()[0]), connectingFrom = derived(() => $$derived_array()[1]), connectingTo = derived(() => $$derived_array()[2]), isPossibleTargetHandle = derived(() => $$derived_array()[3]), valid = derived(() => $$derived_array()[4]);
    $$renderer2.push(`<div${attributes(
      {
        "data-handleid": handleId,
        "data-nodeid": nodeId,
        "data-handlepos": position,
        "data-id": `${stringify(store.flowId)}-${stringify(nodeId)}-${stringify(handleId ?? "null")}-${stringify(type)}`,
        class: clsx([
          "svelte-flow__handle",
          `svelte-flow__handle-${position}`,
          store.noDragClass,
          store.noPanClass,
          position,
          className
        ]),
        style,
        role: "button",
        "aria-label": ariaLabelConfig()[`handle.ariaLabel`],
        tabindex: "-1",
        ...rest
      },
      void 0,
      {
        valid,
        connectingto: connectingTo(),
        connectingfrom: connectingFrom(),
        source: !isTarget(),
        target: isTarget(),
        connectablestart: isConnectableStart,
        connectableend: isConnectableEnd,
        connectable: isConnectable(),
        connectionindicator: isConnectable() && (!connectionInProgress() || isPossibleTargetHandle()) && (connectionInProgress() || store.clickConnectStartHandle ? isConnectableEnd : isConnectableStart)
      }
    )}>`);
    children?.($$renderer2);
    $$renderer2.push(`<!----></div>`);
  });
}
function DefaultNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      data,
      targetPosition = Position.Top,
      sourcePosition = Position.Bottom
    } = $$props;
    Handle($$renderer2, { type: "target", position: targetPosition });
    $$renderer2.push(`<!----> ${escape_html(data?.label)} `);
    Handle($$renderer2, { type: "source", position: sourcePosition });
    $$renderer2.push(`<!---->`);
  });
}
function InputNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { data = { label: "Node" }, sourcePosition = Position.Bottom } = $$props;
    $$renderer2.push(`<!---->${escape_html(data?.label)} `);
    Handle($$renderer2, { type: "source", position: sourcePosition });
    $$renderer2.push(`<!---->`);
  });
}
function OutputNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { data = { label: "Node" }, targetPosition = Position.Top } = $$props;
    $$renderer2.push(`<!---->${escape_html(data?.label)} `);
    Handle($$renderer2, { type: "target", position: targetPosition });
    $$renderer2.push(`<!---->`);
  });
}
function GroupNode($$renderer, $$props) {
}
function hideOnSSR() {
  let hide = typeof window === "undefined";
  return {
    get value() {
      return hide;
    }
  };
}
const isNode = (element) => isNodeBase(element);
const isEdge = (element) => isEdgeBase(element);
function toPxString(value) {
  return value === void 0 ? void 0 : `${value}px`;
}
function EdgeLabel($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      x = 0,
      y = 0,
      width,
      height,
      selectEdgeOnClick = false,
      transparent = false,
      class: className,
      children,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    const store = useStore();
    const edgeId = getEdgeIdContext("EdgeLabel must be used within a Custom Edge component");
    let z = derived(() => {
      return store.visible.edges.get(edgeId)?.zIndex;
    });
    $$renderer2.push(`<div${attributes(
      {
        class: clsx(["svelte-flow__edge-label", { transparent }, className]),
        tabindex: "-1",
        ...rest
      },
      "svelte-qwvnl1",
      void 0,
      {
        display: hideOnSSR().value ? "none" : void 0,
        cursor: selectEdgeOnClick ? "pointer" : void 0,
        transform: `translate(-50%, -50%) translate(${stringify(x)}px,${stringify(y)}px)`,
        "pointer-events": "all",
        width: toPxString(width),
        height: toPxString(height),
        "z-index": z()
      }
    )}>`);
    children?.($$renderer2);
    $$renderer2.push(`<!----></div>`);
  });
}
function BaseEdge($$renderer, $$props) {
  let {
    id,
    path,
    label,
    labelX,
    labelY,
    labelStyle,
    markerStart,
    markerEnd,
    style,
    interactionWidth = 20,
    class: className,
    $$slots,
    $$events,
    ...rest
  } = $$props;
  $$renderer.push(`<path${attr("id", id)}${attr("d", path)}${attr_class(clsx(["svelte-flow__edge-path", className]))}${attr("marker-start", markerStart)}${attr("marker-end", markerEnd)} fill="none"${attr_style(style)}></path>`);
  if (interactionWidth > 0) {
    $$renderer.push("<!--[0-->");
    $$renderer.push(`<path${attributes(
      {
        d: path,
        "stroke-opacity": 0,
        "stroke-width": interactionWidth,
        fill: "none",
        class: "svelte-flow__edge-interaction",
        ...rest
      },
      void 0,
      void 0,
      void 0,
      3
    )}></path>`);
  } else {
    $$renderer.push("<!--[-1-->");
  }
  $$renderer.push(`<!--]-->`);
  if (label) {
    $$renderer.push("<!--[0-->");
    EdgeLabel($$renderer, {
      x: labelX,
      y: labelY,
      style: labelStyle,
      selectEdgeOnClick: true,
      children: ($$renderer2) => {
        $$renderer2.push(`<!---->${escape_html(label)}`);
      },
      $$slots: { default: true }
    });
  } else {
    $$renderer.push("<!--[-1-->");
  }
  $$renderer.push(`<!--]-->`);
}
function BezierEdge($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      id,
      interactionWidth,
      label,
      labelStyle,
      markerEnd,
      markerStart,
      pathOptions,
      sourcePosition,
      sourceX,
      sourceY,
      style,
      targetPosition,
      targetX,
      targetY
    } = $$props;
    let $$d = derived(() => getBezierPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      curvature: pathOptions?.curvature
    })), $$derived_array = derived(() => to_array($$d(), 3)), path = derived(() => $$derived_array()[0]), labelX = derived(() => $$derived_array()[1]), labelY = derived(() => $$derived_array()[2]);
    BaseEdge($$renderer2, {
      id,
      path: path(),
      labelX: labelX(),
      labelY: labelY(),
      label,
      labelStyle,
      markerStart,
      markerEnd,
      interactionWidth,
      style
    });
  });
}
function SmoothStepEdgeInternal($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      interactionWidth,
      label,
      labelStyle,
      style,
      markerEnd,
      markerStart,
      sourcePosition,
      sourceX,
      sourceY,
      targetPosition,
      targetX,
      targetY
    } = $$props;
    let $$d = derived(() => getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition
    })), $$derived_array = derived(() => to_array($$d(), 3)), path = derived(() => $$derived_array()[0]), labelX = derived(() => $$derived_array()[1]), labelY = derived(() => $$derived_array()[2]);
    BaseEdge($$renderer2, {
      path: path(),
      labelX: labelX(),
      labelY: labelY(),
      label,
      labelStyle,
      markerStart,
      markerEnd,
      interactionWidth,
      style
    });
  });
}
function StraightEdgeInternal($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      sourceX,
      sourceY,
      targetX,
      targetY,
      label,
      labelStyle,
      markerStart,
      markerEnd,
      interactionWidth,
      style
    } = $$props;
    let $$d = derived(() => getStraightPath({ sourceX, sourceY, targetX, targetY })), $$derived_array = derived(() => to_array($$d(), 3)), path = derived(() => $$derived_array()[0]), labelX = derived(() => $$derived_array()[1]), labelY = derived(() => $$derived_array()[2]);
    BaseEdge($$renderer2, {
      path: path(),
      labelX: labelX(),
      labelY: labelY(),
      label,
      labelStyle,
      markerStart,
      markerEnd,
      interactionWidth,
      style
    });
  });
}
function StepEdgeInternal($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      label,
      labelStyle,
      markerStart,
      markerEnd,
      interactionWidth,
      style
    } = $$props;
    let $$d = derived(() => getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 0
    })), $$derived_array = derived(() => to_array($$d(), 3)), path = derived(() => $$derived_array()[0]), labelX = derived(() => $$derived_array()[1]), labelY = derived(() => $$derived_array()[2]);
    BaseEdge($$renderer2, {
      path: path(),
      labelX: labelX(),
      labelY: labelY(),
      label,
      labelStyle,
      markerStart,
      markerEnd,
      interactionWidth,
      style
    });
  });
}
function getVisibleNodes(nodeLookup, transform, width, height) {
  const visibleNodes = /* @__PURE__ */ new Map();
  getNodesInside(nodeLookup, { x: 0, y: 0, width, height }, transform, true).forEach((node) => {
    visibleNodes.set(node.id, node);
  });
  return visibleNodes;
}
function getLayoutedEdges(options) {
  const { edges, defaultEdgeOptions, nodeLookup, previousEdges, connectionMode, onerror, onlyRenderVisible, elevateEdgesOnSelect, zIndexMode } = options;
  const layoutedEdges = /* @__PURE__ */ new Map();
  for (const edge of edges) {
    const sourceNode = nodeLookup.get(edge.source);
    const targetNode = nodeLookup.get(edge.target);
    if (!sourceNode || !targetNode) {
      continue;
    }
    if (onlyRenderVisible) {
      const { visibleNodes, transform, width, height } = options;
      if (isEdgeVisible({
        sourceNode,
        targetNode,
        width,
        height,
        transform
      })) {
        visibleNodes.set(sourceNode.id, sourceNode);
        visibleNodes.set(targetNode.id, targetNode);
      } else {
        continue;
      }
    }
    const previous = previousEdges.get(edge.id);
    if (previous && edge === previous.edge && sourceNode == previous.sourceNode && targetNode == previous.targetNode) {
      layoutedEdges.set(edge.id, previous);
      continue;
    }
    const edgePosition = getEdgePosition({
      id: edge.id,
      sourceNode,
      targetNode,
      sourceHandle: edge.sourceHandle || null,
      targetHandle: edge.targetHandle || null,
      connectionMode,
      onError: onerror
    });
    if (edgePosition) {
      layoutedEdges.set(edge.id, {
        ...defaultEdgeOptions,
        ...edge,
        ...edgePosition,
        zIndex: getElevatedEdgeZIndex({
          selected: edge.selected,
          zIndex: edge.zIndex ?? defaultEdgeOptions.zIndex,
          sourceNode,
          targetNode,
          elevateOnSelect: elevateEdgesOnSelect,
          zIndexMode
        }),
        sourceNode,
        targetNode,
        edge
      });
    }
  }
  return layoutedEdges;
}
const devWarn = createDevWarn("Svelte Flow", "https://svelteflow.dev/");
const initialNodeTypes = {
  input: InputNode,
  output: OutputNode,
  default: DefaultNode,
  group: GroupNode
};
const initialEdgeTypes = {
  straight: StraightEdgeInternal,
  smoothstep: SmoothStepEdgeInternal,
  default: BezierEdge,
  step: StepEdgeInternal
};
function getInitialViewport(_nodesInitialized, fitView, initialViewport, width, height, nodeLookup) {
  if (fitView && !initialViewport && width && height) {
    const bounds = getInternalNodesBounds(nodeLookup, {
      filter: (node) => !!((node.width || node.initialWidth) && (node.height || node.initialHeight))
    });
    return getViewportForBounds(bounds, width, height, 0.5, 2, 0.1);
  } else {
    return initialViewport ?? { x: 0, y: 0, zoom: 1 };
  }
}
function getInitialStore(signals) {
  class SvelteFlowStore {
    #flowId = derived(() => signals.props.id ?? "1");
    get flowId() {
      return this.#flowId();
    }
    set flowId($$value) {
      return this.#flowId($$value);
    }
    domNode = null;
    panZoom = null;
    width = signals.width ?? 0;
    height = signals.height ?? 0;
    zIndexMode = signals.props.zIndexMode ?? "basic";
    #nodesInitialized = derived(() => {
      const { nodesInitialized } = adoptUserNodes(signals.nodes, this.nodeLookup, this.parentLookup, {
        nodeExtent: this.nodeExtent,
        nodeOrigin: this.nodeOrigin,
        elevateNodesOnSelect: signals.props.elevateNodesOnSelect ?? true,
        checkEquality: true,
        zIndexMode: this.zIndexMode
      });
      if (this.fitViewQueued && nodesInitialized) {
        if (this.fitViewOptions?.duration) {
          this.resolveFitView();
        } else {
          queueMicrotask(() => {
            this.resolveFitView();
          });
        }
      }
      return nodesInitialized;
    });
    get nodesInitialized() {
      return this.#nodesInitialized();
    }
    set nodesInitialized($$value) {
      return this.#nodesInitialized($$value);
    }
    #viewportInitialized = derived(() => this.panZoom !== null);
    get viewportInitialized() {
      return this.#viewportInitialized();
    }
    set viewportInitialized($$value) {
      return this.#viewportInitialized($$value);
    }
    #_edges = derived(() => {
      updateConnectionLookup(this.connectionLookup, this.edgeLookup, signals.edges);
      return signals.edges;
    });
    get _edges() {
      return this.#_edges();
    }
    set _edges($$value) {
      return this.#_edges($$value);
    }
    get nodes() {
      this.nodesInitialized;
      return signals.nodes;
    }
    set nodes(nodes) {
      signals.nodes = nodes;
    }
    get edges() {
      return this._edges;
    }
    set edges(edges) {
      signals.edges = edges;
    }
    _prevSelectedNodes = [];
    _prevSelectedNodeIds = /* @__PURE__ */ new Set();
    #selectedNodes = derived(() => {
      const selectedNodesCount = this._prevSelectedNodeIds.size;
      const selectedNodeIds = /* @__PURE__ */ new Set();
      const selectedNodes = this.nodes.filter((node) => {
        if (node.selected) {
          selectedNodeIds.add(node.id);
          this._prevSelectedNodeIds.delete(node.id);
        }
        return node.selected;
      });
      if (selectedNodesCount !== selectedNodeIds.size || this._prevSelectedNodeIds.size > 0) {
        this._prevSelectedNodes = selectedNodes;
      }
      this._prevSelectedNodeIds = selectedNodeIds;
      return this._prevSelectedNodes;
    });
    get selectedNodes() {
      return this.#selectedNodes();
    }
    set selectedNodes($$value) {
      return this.#selectedNodes($$value);
    }
    _prevSelectedEdges = [];
    _prevSelectedEdgeIds = /* @__PURE__ */ new Set();
    #selectedEdges = derived(() => {
      const selectedEdgesCount = this._prevSelectedEdgeIds.size;
      const selectedEdgeIds = /* @__PURE__ */ new Set();
      const selectedEdges = this.edges.filter((edge) => {
        if (edge.selected) {
          selectedEdgeIds.add(edge.id);
          this._prevSelectedEdgeIds.delete(edge.id);
        }
        return edge.selected;
      });
      if (selectedEdgesCount !== selectedEdgeIds.size || this._prevSelectedEdgeIds.size > 0) {
        this._prevSelectedEdges = selectedEdges;
      }
      this._prevSelectedEdgeIds = selectedEdgeIds;
      return this._prevSelectedEdges;
    });
    get selectedEdges() {
      return this.#selectedEdges();
    }
    set selectedEdges($$value) {
      return this.#selectedEdges($$value);
    }
    selectionChangeHandlers = /* @__PURE__ */ new Map();
    nodeLookup = /* @__PURE__ */ new Map();
    parentLookup = /* @__PURE__ */ new Map();
    connectionLookup = /* @__PURE__ */ new Map();
    edgeLookup = /* @__PURE__ */ new Map();
    _prevVisibleEdges = /* @__PURE__ */ new Map();
    #visible = derived(() => {
      const {
        // We need to access this._nodes to trigger on changes
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        nodes,
        _edges: edges,
        _prevVisibleEdges: previousEdges,
        nodeLookup,
        connectionMode,
        onerror,
        onlyRenderVisibleElements,
        defaultEdgeOptions,
        zIndexMode
      } = this;
      let visibleNodes;
      let visibleEdges;
      const options = {
        edges,
        defaultEdgeOptions,
        previousEdges,
        nodeLookup,
        connectionMode,
        elevateEdgesOnSelect: signals.props.elevateEdgesOnSelect ?? true,
        zIndexMode,
        onerror
      };
      if (onlyRenderVisibleElements) {
        const { viewport, width, height } = this;
        const transform = [viewport.x, viewport.y, viewport.zoom];
        visibleNodes = getVisibleNodes(nodeLookup, transform, width, height);
        visibleEdges = getLayoutedEdges({
          ...options,
          onlyRenderVisible: true,
          visibleNodes,
          transform,
          width,
          height
        });
      } else {
        visibleNodes = this.nodeLookup;
        visibleEdges = getLayoutedEdges(options);
      }
      return { nodes: visibleNodes, edges: visibleEdges };
    });
    get visible() {
      return this.#visible();
    }
    set visible($$value) {
      return this.#visible($$value);
    }
    #nodesDraggable = derived(() => signals.props.nodesDraggable ?? true);
    get nodesDraggable() {
      return this.#nodesDraggable();
    }
    set nodesDraggable($$value) {
      return this.#nodesDraggable($$value);
    }
    #nodesConnectable = derived(() => signals.props.nodesConnectable ?? true);
    get nodesConnectable() {
      return this.#nodesConnectable();
    }
    set nodesConnectable($$value) {
      return this.#nodesConnectable($$value);
    }
    #elementsSelectable = derived(() => signals.props.elementsSelectable ?? true);
    get elementsSelectable() {
      return this.#elementsSelectable();
    }
    set elementsSelectable($$value) {
      return this.#elementsSelectable($$value);
    }
    #nodesFocusable = derived(() => signals.props.nodesFocusable ?? true);
    get nodesFocusable() {
      return this.#nodesFocusable();
    }
    set nodesFocusable($$value) {
      return this.#nodesFocusable($$value);
    }
    #edgesFocusable = derived(() => signals.props.edgesFocusable ?? true);
    get edgesFocusable() {
      return this.#edgesFocusable();
    }
    set edgesFocusable($$value) {
      return this.#edgesFocusable($$value);
    }
    #disableKeyboardA11y = derived(() => signals.props.disableKeyboardA11y ?? false);
    get disableKeyboardA11y() {
      return this.#disableKeyboardA11y();
    }
    set disableKeyboardA11y($$value) {
      return this.#disableKeyboardA11y($$value);
    }
    #minZoom = derived(() => signals.props.minZoom ?? 0.5);
    get minZoom() {
      return this.#minZoom();
    }
    set minZoom($$value) {
      return this.#minZoom($$value);
    }
    #maxZoom = derived(() => signals.props.maxZoom ?? 2);
    get maxZoom() {
      return this.#maxZoom();
    }
    set maxZoom($$value) {
      return this.#maxZoom($$value);
    }
    #nodeOrigin = derived(() => signals.props.nodeOrigin ?? [0, 0]);
    get nodeOrigin() {
      return this.#nodeOrigin();
    }
    set nodeOrigin($$value) {
      return this.#nodeOrigin($$value);
    }
    #nodeExtent = derived(() => signals.props.nodeExtent ?? infiniteExtent);
    get nodeExtent() {
      return this.#nodeExtent();
    }
    set nodeExtent($$value) {
      return this.#nodeExtent($$value);
    }
    #translateExtent = derived(() => signals.props.translateExtent ?? infiniteExtent);
    get translateExtent() {
      return this.#translateExtent();
    }
    set translateExtent($$value) {
      return this.#translateExtent($$value);
    }
    #defaultEdgeOptions = derived(() => signals.props.defaultEdgeOptions ?? {});
    get defaultEdgeOptions() {
      return this.#defaultEdgeOptions();
    }
    set defaultEdgeOptions($$value) {
      return this.#defaultEdgeOptions($$value);
    }
    #nodeDragThreshold = derived(() => signals.props.nodeDragThreshold ?? 1);
    get nodeDragThreshold() {
      return this.#nodeDragThreshold();
    }
    set nodeDragThreshold($$value) {
      return this.#nodeDragThreshold($$value);
    }
    #autoPanOnNodeDrag = derived(() => signals.props.autoPanOnNodeDrag ?? true);
    get autoPanOnNodeDrag() {
      return this.#autoPanOnNodeDrag();
    }
    set autoPanOnNodeDrag($$value) {
      return this.#autoPanOnNodeDrag($$value);
    }
    #autoPanOnConnect = derived(() => signals.props.autoPanOnConnect ?? true);
    get autoPanOnConnect() {
      return this.#autoPanOnConnect();
    }
    set autoPanOnConnect($$value) {
      return this.#autoPanOnConnect($$value);
    }
    #autoPanOnNodeFocus = derived(() => signals.props.autoPanOnNodeFocus ?? true);
    get autoPanOnNodeFocus() {
      return this.#autoPanOnNodeFocus();
    }
    set autoPanOnNodeFocus($$value) {
      return this.#autoPanOnNodeFocus($$value);
    }
    #autoPanSpeed = derived(() => signals.props.autoPanSpeed ?? 15);
    get autoPanSpeed() {
      return this.#autoPanSpeed();
    }
    set autoPanSpeed($$value) {
      return this.#autoPanSpeed($$value);
    }
    #connectionDragThreshold = derived(() => signals.props.connectionDragThreshold ?? 1);
    get connectionDragThreshold() {
      return this.#connectionDragThreshold();
    }
    set connectionDragThreshold($$value) {
      return this.#connectionDragThreshold($$value);
    }
    fitViewQueued = signals.props.fitView ?? false;
    fitViewOptions = signals.props.fitViewOptions;
    fitViewResolver = null;
    #snapGrid = derived(() => signals.props.snapGrid ?? null);
    get snapGrid() {
      return this.#snapGrid();
    }
    set snapGrid($$value) {
      return this.#snapGrid($$value);
    }
    dragging = false;
    selectionRect = null;
    selectionKeyPressed = false;
    multiselectionKeyPressed = false;
    deleteKeyPressed = false;
    panActivationKeyPressed = false;
    zoomActivationKeyPressed = false;
    selectionRectMode = null;
    ariaLiveMessage = "";
    #selectionMode = derived(() => signals.props.selectionMode ?? SelectionMode.Partial);
    get selectionMode() {
      return this.#selectionMode();
    }
    set selectionMode($$value) {
      return this.#selectionMode($$value);
    }
    #nodeTypes = derived(() => ({ ...initialNodeTypes, ...signals.props.nodeTypes }));
    get nodeTypes() {
      return this.#nodeTypes();
    }
    set nodeTypes($$value) {
      return this.#nodeTypes($$value);
    }
    #edgeTypes = derived(() => ({ ...initialEdgeTypes, ...signals.props.edgeTypes }));
    get edgeTypes() {
      return this.#edgeTypes();
    }
    set edgeTypes($$value) {
      return this.#edgeTypes($$value);
    }
    #noPanClass = derived(() => signals.props.noPanClass ?? "nopan");
    get noPanClass() {
      return this.#noPanClass();
    }
    set noPanClass($$value) {
      return this.#noPanClass($$value);
    }
    #noDragClass = derived(() => signals.props.noDragClass ?? "nodrag");
    get noDragClass() {
      return this.#noDragClass();
    }
    set noDragClass($$value) {
      return this.#noDragClass($$value);
    }
    #noWheelClass = derived(() => signals.props.noWheelClass ?? "nowheel");
    get noWheelClass() {
      return this.#noWheelClass();
    }
    set noWheelClass($$value) {
      return this.#noWheelClass($$value);
    }
    #ariaLabelConfig = derived(() => mergeAriaLabelConfig(signals.props.ariaLabelConfig));
    get ariaLabelConfig() {
      return this.#ariaLabelConfig();
    }
    set ariaLabelConfig($$value) {
      return this.#ariaLabelConfig($$value);
    }
    _viewport = getInitialViewport(this.nodesInitialized, signals.props.fitView, signals.props.initialViewport, this.width, this.height, this.nodeLookup);
    get viewport() {
      return signals.viewport ?? this._viewport;
    }
    set viewport(newViewport) {
      if (signals.viewport) {
        signals.viewport = newViewport;
      }
      this._viewport = newViewport;
    }
    // _connection is viewport independent and originating from XYHandle
    _connection = initialConnection;
    #connection = derived(
      // We derive a viewport dependent connection here
      () => {
        if (!this._connection.inProgress) {
          return this._connection;
        }
        return {
          ...this._connection,
          to: pointToRendererPoint(this._connection.to, [this.viewport.x, this.viewport.y, this.viewport.zoom])
        };
      }
    );
    get connection() {
      return this.#connection();
    }
    set connection($$value) {
      return this.#connection($$value);
    }
    #connectionMode = derived(() => signals.props.connectionMode ?? ConnectionMode.Strict);
    get connectionMode() {
      return this.#connectionMode();
    }
    set connectionMode($$value) {
      return this.#connectionMode($$value);
    }
    #connectionRadius = derived(() => signals.props.connectionRadius ?? 20);
    get connectionRadius() {
      return this.#connectionRadius();
    }
    set connectionRadius($$value) {
      return this.#connectionRadius($$value);
    }
    #isValidConnection = derived(() => signals.props.isValidConnection ?? (() => true));
    get isValidConnection() {
      return this.#isValidConnection();
    }
    set isValidConnection($$value) {
      return this.#isValidConnection($$value);
    }
    #selectNodesOnDrag = derived(() => signals.props.selectNodesOnDrag ?? true);
    get selectNodesOnDrag() {
      return this.#selectNodesOnDrag();
    }
    set selectNodesOnDrag($$value) {
      return this.#selectNodesOnDrag($$value);
    }
    #defaultMarkerColor = derived(() => signals.props.defaultMarkerColor === void 0 ? "#b1b1b7" : signals.props.defaultMarkerColor);
    get defaultMarkerColor() {
      return this.#defaultMarkerColor();
    }
    set defaultMarkerColor($$value) {
      return this.#defaultMarkerColor($$value);
    }
    #markers = derived(() => {
      return createMarkerIds(signals.edges, {
        defaultColor: this.defaultMarkerColor,
        id: this.flowId,
        defaultMarkerStart: this.defaultEdgeOptions.markerStart,
        defaultMarkerEnd: this.defaultEdgeOptions.markerEnd
      });
    });
    get markers() {
      return this.#markers();
    }
    set markers($$value) {
      return this.#markers($$value);
    }
    #onlyRenderVisibleElements = derived(() => signals.props.onlyRenderVisibleElements ?? false);
    get onlyRenderVisibleElements() {
      return this.#onlyRenderVisibleElements();
    }
    set onlyRenderVisibleElements($$value) {
      return this.#onlyRenderVisibleElements($$value);
    }
    #onerror = derived(() => signals.props.onflowerror ?? devWarn);
    get onerror() {
      return this.#onerror();
    }
    set onerror($$value) {
      return this.#onerror($$value);
    }
    #ondelete = derived(() => signals.props.ondelete);
    get ondelete() {
      return this.#ondelete();
    }
    set ondelete($$value) {
      return this.#ondelete($$value);
    }
    #onbeforedelete = derived(() => signals.props.onbeforedelete);
    get onbeforedelete() {
      return this.#onbeforedelete();
    }
    set onbeforedelete($$value) {
      return this.#onbeforedelete($$value);
    }
    #onbeforeconnect = derived(() => signals.props.onbeforeconnect);
    get onbeforeconnect() {
      return this.#onbeforeconnect();
    }
    set onbeforeconnect($$value) {
      return this.#onbeforeconnect($$value);
    }
    #onconnect = derived(() => signals.props.onconnect);
    get onconnect() {
      return this.#onconnect();
    }
    set onconnect($$value) {
      return this.#onconnect($$value);
    }
    #onconnectstart = derived(() => signals.props.onconnectstart);
    get onconnectstart() {
      return this.#onconnectstart();
    }
    set onconnectstart($$value) {
      return this.#onconnectstart($$value);
    }
    #onconnectend = derived(() => signals.props.onconnectend);
    get onconnectend() {
      return this.#onconnectend();
    }
    set onconnectend($$value) {
      return this.#onconnectend($$value);
    }
    #onbeforereconnect = derived(() => signals.props.onbeforereconnect);
    get onbeforereconnect() {
      return this.#onbeforereconnect();
    }
    set onbeforereconnect($$value) {
      return this.#onbeforereconnect($$value);
    }
    #onreconnect = derived(() => signals.props.onreconnect);
    get onreconnect() {
      return this.#onreconnect();
    }
    set onreconnect($$value) {
      return this.#onreconnect($$value);
    }
    #onreconnectstart = derived(() => signals.props.onreconnectstart);
    get onreconnectstart() {
      return this.#onreconnectstart();
    }
    set onreconnectstart($$value) {
      return this.#onreconnectstart($$value);
    }
    #onreconnectend = derived(() => signals.props.onreconnectend);
    get onreconnectend() {
      return this.#onreconnectend();
    }
    set onreconnectend($$value) {
      return this.#onreconnectend($$value);
    }
    #clickConnect = derived(() => signals.props.clickConnect ?? true);
    get clickConnect() {
      return this.#clickConnect();
    }
    set clickConnect($$value) {
      return this.#clickConnect($$value);
    }
    #onclickconnectstart = derived(() => signals.props.onclickconnectstart);
    get onclickconnectstart() {
      return this.#onclickconnectstart();
    }
    set onclickconnectstart($$value) {
      return this.#onclickconnectstart($$value);
    }
    #onclickconnectend = derived(() => signals.props.onclickconnectend);
    get onclickconnectend() {
      return this.#onclickconnectend();
    }
    set onclickconnectend($$value) {
      return this.#onclickconnectend($$value);
    }
    clickConnectStartHandle = null;
    #onselectiondrag = derived(() => signals.props.onselectiondrag);
    get onselectiondrag() {
      return this.#onselectiondrag();
    }
    set onselectiondrag($$value) {
      return this.#onselectiondrag($$value);
    }
    #onselectiondragstart = derived(() => signals.props.onselectiondragstart);
    get onselectiondragstart() {
      return this.#onselectiondragstart();
    }
    set onselectiondragstart($$value) {
      return this.#onselectiondragstart($$value);
    }
    #onselectiondragstop = derived(() => signals.props.onselectiondragstop);
    get onselectiondragstop() {
      return this.#onselectiondragstop();
    }
    set onselectiondragstop($$value) {
      return this.#onselectiondragstop($$value);
    }
    resolveFitView = async () => {
      if (!this.panZoom) {
        return;
      }
      await fitViewport(
        {
          nodes: this.nodeLookup,
          width: this.width,
          height: this.height,
          panZoom: this.panZoom,
          minZoom: this.minZoom,
          maxZoom: this.maxZoom
        },
        this.fitViewOptions
      );
      this.fitViewResolver?.resolve(true);
      this.fitViewQueued = false;
      this.fitViewOptions = void 0;
      this.fitViewResolver = null;
    };
    _prefersDark = new MediaQuery("(prefers-color-scheme: dark)", signals.props.colorModeSSR === "dark");
    #colorMode = derived(() => signals.props.colorMode === "system" ? this._prefersDark.current ? "dark" : "light" : signals.props.colorMode ?? "light");
    get colorMode() {
      return this.#colorMode();
    }
    set colorMode($$value) {
      return this.#colorMode($$value);
    }
    constructor() {
      if (process.env.NODE_ENV === "development") {
        warnIfDeeplyReactive(signals.nodes, "nodes");
        warnIfDeeplyReactive(signals.edges, "edges");
      }
    }
    resetStoreValues() {
      this.dragging = false;
      this.selectionRect = null;
      this.selectionRectMode = null;
      this.selectionKeyPressed = false;
      this.multiselectionKeyPressed = false;
      this.deleteKeyPressed = false;
      this.panActivationKeyPressed = false;
      this.zoomActivationKeyPressed = false;
      this._connection = initialConnection;
      this.clickConnectStartHandle = null;
      this.viewport = signals.props.initialViewport ?? { x: 0, y: 0, zoom: 1 };
      this.ariaLiveMessage = "";
    }
  }
  return new SvelteFlowStore();
}
function warnIfDeeplyReactive(array, name) {
  try {
    if (array && array.length > 0) {
      structuredClone(array[0]);
    }
  } catch {
    console.warn(`Use $state.raw for ${name} to prevent performance issues.`);
  }
}
const providerErrorMessage = errorMessages["error001"]("svelte");
function useStore() {
  const storeContext = getContext(key);
  if (!storeContext) {
    throw new Error(providerErrorMessage);
  }
  return storeContext.getStore();
}
const key = /* @__PURE__ */ Symbol();
function createStore(signals) {
  const store = getInitialStore(signals);
  function setNodeTypes(nodeTypes) {
    store.nodeTypes = {
      ...initialNodeTypes,
      ...nodeTypes
    };
  }
  function setEdgeTypes(edgeTypes) {
    store.edgeTypes = {
      ...initialEdgeTypes,
      ...edgeTypes
    };
  }
  function addEdge$12(edgeParams) {
    store.edges = addEdge(edgeParams, store.edges, { onError: store.onerror });
  }
  const updateNodePositions = (nodeDragItems, dragging = false) => {
    store.nodes = store.nodes.map((node) => {
      if (store.connection.inProgress && store.connection.fromNode.id === node.id) {
        const internalNode = store.nodeLookup.get(node.id);
        if (internalNode) {
          store.connection = {
            ...store.connection,
            from: getHandlePosition(internalNode, store.connection.fromHandle, Position.Left, true)
          };
        }
      }
      const dragItem = nodeDragItems.get(node.id);
      return dragItem ? { ...node, position: dragItem.position, dragging } : node;
    });
  };
  function updateNodeInternals$1(updates) {
    const { changes, updatedInternals } = updateNodeInternals(updates, store.nodeLookup, store.parentLookup, store.domNode, store.nodeOrigin, store.nodeExtent, store.zIndexMode);
    if (!updatedInternals) {
      return;
    }
    updateAbsolutePositions(store.nodeLookup, store.parentLookup, {
      nodeOrigin: store.nodeOrigin,
      nodeExtent: store.nodeExtent,
      zIndexMode: store.zIndexMode
    });
    if (store.fitViewQueued) {
      store.resolveFitView();
    }
    const newNodes = /* @__PURE__ */ new Map();
    for (const change of changes) {
      const userNode = store.nodeLookup.get(change.id)?.internals.userNode;
      if (!userNode) {
        continue;
      }
      const node = { ...userNode };
      switch (change.type) {
        case "dimensions": {
          const measured = { ...node.measured, ...change.dimensions };
          if (change.setAttributes) {
            node.width = change.dimensions?.width ?? node.width;
            node.height = change.dimensions?.height ?? node.height;
          }
          node.measured = measured;
          break;
        }
        case "position":
          node.position = change.position ?? node.position;
          break;
      }
      newNodes.set(change.id, node);
    }
    store.nodes = store.nodes.map((node) => newNodes.get(node.id) ?? node);
  }
  function fitView(options) {
    const fitViewResolver = store.fitViewResolver ?? Promise.withResolvers();
    store.fitViewQueued = true;
    store.fitViewOptions = options;
    store.fitViewResolver = fitViewResolver;
    store.nodes = [...store.nodes];
    return fitViewResolver.promise;
  }
  async function setCenter(x, y, options) {
    const nextZoom = typeof options?.zoom !== "undefined" ? options.zoom : store.maxZoom;
    const currentPanZoom = store.panZoom;
    if (!currentPanZoom) {
      return false;
    }
    await currentPanZoom.setViewport({
      x: store.width / 2 - x * nextZoom,
      y: store.height / 2 - y * nextZoom,
      zoom: nextZoom
    }, { duration: options?.duration, ease: options?.ease, interpolate: options?.interpolate });
    return true;
  }
  async function zoomBy(factor, options) {
    const panZoom = store.panZoom;
    if (!panZoom) {
      return false;
    }
    return panZoom.scaleBy(factor, options);
  }
  async function zoomIn(options) {
    return zoomBy(1.2, options);
  }
  function zoomOut(options) {
    return zoomBy(1 / 1.2, options);
  }
  function setMinZoom(minZoom) {
    const panZoom = store.panZoom;
    if (panZoom) {
      panZoom.setScaleExtent([minZoom, store.maxZoom]);
      store.minZoom = minZoom;
    }
  }
  function setMaxZoom(maxZoom) {
    const panZoom = store.panZoom;
    if (panZoom) {
      panZoom.setScaleExtent([store.minZoom, maxZoom]);
      store.maxZoom = maxZoom;
    }
  }
  function setTranslateExtent(extent) {
    const panZoom = store.panZoom;
    if (panZoom) {
      panZoom.setTranslateExtent(extent);
      store.translateExtent = extent;
    }
  }
  function deselect(elements, elementsToDeselect = null) {
    let deselected = false;
    const newElements = elements.map((element) => {
      const shouldDeselect = elementsToDeselect ? elementsToDeselect.has(element.id) : true;
      if (shouldDeselect && element.selected) {
        deselected = true;
        return { ...element, selected: false };
      }
      return element;
    });
    return [deselected, newElements];
  }
  function unselectNodesAndEdges(params) {
    const nodesToDeselect = params?.nodes ? new Set(params.nodes.map((node) => node.id)) : null;
    const [nodesDeselected, newNodes] = deselect(store.nodes, nodesToDeselect);
    if (nodesDeselected) {
      store.nodes = newNodes;
    }
    const edgesToDeselect = params?.edges ? new Set(params.edges.map((node) => node.id)) : null;
    const [edgesDeselected, newEdges] = deselect(store.edges, edgesToDeselect);
    if (edgesDeselected) {
      store.edges = newEdges;
    }
  }
  function addSelectedNodes(ids) {
    const isMultiSelection = store.multiselectionKeyPressed;
    store.nodes = store.nodes.map((node) => {
      const nodeWillBeSelected = ids.includes(node.id);
      const selected = isMultiSelection ? node.selected || nodeWillBeSelected : nodeWillBeSelected;
      if (!!node.selected !== selected) {
        return { ...node, selected };
      }
      return node;
    });
    if (!isMultiSelection) {
      unselectNodesAndEdges({ nodes: [] });
    }
  }
  function addSelectedEdges(ids) {
    const isMultiSelection = store.multiselectionKeyPressed;
    store.edges = store.edges.map((edge) => {
      const edgeWillBeSelected = ids.includes(edge.id);
      const selected = isMultiSelection ? edge.selected || edgeWillBeSelected : edgeWillBeSelected;
      if (!!edge.selected !== selected) {
        return { ...edge, selected };
      }
      return edge;
    });
    if (!isMultiSelection) {
      unselectNodesAndEdges({ edges: [] });
    }
  }
  function handleNodeSelection(id, unselect, nodeRef) {
    const node = store.nodeLookup.get(id);
    if (!node) {
      store.onerror("012", errorMessages["error012"](id));
      return;
    }
    store.selectionRect = null;
    store.selectionRectMode = null;
    if (!node.selected) {
      addSelectedNodes([id]);
    } else if (unselect || node.selected && store.multiselectionKeyPressed) {
      unselectNodesAndEdges({ nodes: [node.internals.userNode], edges: [] });
      requestAnimationFrame(() => nodeRef?.blur());
    }
  }
  function handleEdgeSelection(id) {
    const edge = store.edgeLookup.get(id);
    if (!edge) {
      store.onerror("016", errorMessages["error016"](id));
      return;
    }
    const selectable = edge.selectable || store.elementsSelectable && typeof edge.selectable === "undefined";
    if (selectable) {
      store.selectionRect = null;
      store.selectionRectMode = null;
      if (!edge.selected) {
        addSelectedEdges([id]);
      } else if (edge.selected && store.multiselectionKeyPressed) {
        unselectNodesAndEdges({ nodes: [], edges: [edge] });
      }
    }
  }
  function moveSelectedNodes(direction, factor) {
    const { nodeExtent, snapGrid, nodeOrigin, nodeLookup, nodesDraggable, onerror } = store;
    const nodeUpdates = /* @__PURE__ */ new Map();
    const xVelo = snapGrid?.[0] ?? 5;
    const yVelo = snapGrid?.[1] ?? 5;
    const xDiff = direction.x * xVelo * factor;
    const yDiff = direction.y * yVelo * factor;
    for (const node of nodeLookup.values()) {
      const isSelected = node.selected && (node.draggable || nodesDraggable && typeof node.draggable === "undefined");
      if (!isSelected) {
        continue;
      }
      let nextPosition = {
        x: node.internals.positionAbsolute.x + xDiff,
        y: node.internals.positionAbsolute.y + yDiff
      };
      if (snapGrid) {
        nextPosition = snapPosition(nextPosition, snapGrid);
      }
      const { position, positionAbsolute } = calculateNodePosition({
        nodeId: node.id,
        nextPosition,
        nodeLookup,
        nodeExtent,
        nodeOrigin,
        onError: onerror
      });
      node.position = position;
      node.internals.positionAbsolute = positionAbsolute;
      nodeUpdates.set(node.id, node);
    }
    updateNodePositions(nodeUpdates);
  }
  function panBy$1(delta) {
    return panBy({
      delta,
      panZoom: store.panZoom,
      transform: [store.viewport.x, store.viewport.y, store.viewport.zoom],
      translateExtent: store.translateExtent,
      width: store.width,
      height: store.height
    });
  }
  const updateConnection = (newConnection) => {
    store._connection = { ...newConnection };
  };
  function cancelConnection() {
    store._connection = initialConnection;
  }
  function reset() {
    store.resetStoreValues();
    unselectNodesAndEdges();
  }
  const storeWithActions = Object.assign(store, {
    setNodeTypes,
    setEdgeTypes,
    addEdge: addEdge$12,
    updateNodePositions,
    updateNodeInternals: updateNodeInternals$1,
    zoomIn,
    zoomOut,
    fitView,
    setCenter,
    setMinZoom,
    setMaxZoom,
    setTranslateExtent,
    unselectNodesAndEdges,
    addSelectedNodes,
    addSelectedEdges,
    handleNodeSelection,
    handleEdgeSelection,
    moveSelectedNodes,
    panBy: panBy$1,
    updateConnection,
    cancelConnection,
    reset
  });
  return storeWithActions;
}
function Zoom($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      store = void 0,
      panOnScrollMode,
      preventScrolling,
      zoomOnScroll,
      zoomOnDoubleClick,
      zoomOnPinch,
      panOnDrag,
      panOnScroll,
      panOnScrollSpeed,
      paneClickDistance,
      selectionOnDrag,
      onmovestart,
      onmove,
      onmoveend,
      oninit,
      children
    } = $$props;
    const { viewport: initialViewport } = store;
    $$renderer2.push(`<div class="svelte-flow__zoom svelte-flow__container">`);
    children($$renderer2);
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { store });
  });
}
function Pane($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      store = void 0,
      panOnDrag = true,
      paneClickDistance = 1,
      selectionOnDrag,
      autoPanOnSelection = true,
      onpaneclick,
      onpanecontextmenu,
      onselectionstart,
      onselectionend,
      children
    } = $$props;
    let panOnDragActive = derived(() => store.panActivationKeyPressed || panOnDrag);
    let isSelecting = derived(() => store.selectionKeyPressed || !!store.selectionRect || selectionOnDrag && panOnDragActive() !== true);
    let autoPanId = 0;
    function cleanupAutoPan() {
      cancelAnimationFrame(autoPanId);
      autoPanId = 0;
    }
    onDestroy(() => {
      if (typeof window !== "undefined") {
        cleanupAutoPan();
      }
    });
    $$renderer2.push(`<div${attr_class("svelte-flow__pane svelte-flow__container", void 0, {
      "draggable": panOnDrag === true || Array.isArray(panOnDrag) && panOnDrag.includes(0),
      "dragging": store.dragging,
      "selection": isSelecting()
    })}>`);
    children($$renderer2);
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { store });
  });
}
function Viewport($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { store = void 0, children } = $$props;
    $$renderer2.push(`<div class="svelte-flow__viewport xyflow__viewport svelte-flow__container"${attr_style("", {
      transform: `translate(${stringify(store.viewport.x)}px, ${stringify(store.viewport.y)}px) scale(${stringify(store.viewport.zoom)})`
    })}>`);
    children($$renderer2);
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { store });
  });
}
function A11yDescriptions($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { store } = $$props;
    $$renderer2.push(`<div${attr("id", `${ARIA_NODE_DESC_KEY}-${store.flowId}`)} class="a11y-hidden svelte-ci89z5">${escape_html(store.disableKeyboardA11y ? store.ariaLabelConfig["node.a11yDescription.default"] : store.ariaLabelConfig["node.a11yDescription.keyboardDisabled"])}</div> <div${attr("id", `${ARIA_EDGE_DESC_KEY}-${store.flowId}`)} class="a11y-hidden svelte-ci89z5">${escape_html(store.ariaLabelConfig["edge.a11yDescription.default"])}</div> `);
    if (!store.disableKeyboardA11y) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div${attr("id", `${ARIA_LIVE_MESSAGE}-${store.flowId}`)} aria-live="assertive" aria-atomic="true" class="a11y-live-msg svelte-ci89z5">${escape_html(store.ariaLiveMessage)}</div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
const ARIA_NODE_DESC_KEY = "svelte-flow__node-desc";
const ARIA_EDGE_DESC_KEY = "svelte-flow__edge-desc";
const ARIA_LIVE_MESSAGE = "svelte-flow__aria-live";
function NodeWrapper($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      store = void 0,
      node,
      resizeObserver,
      nodeClickDistance,
      onnodeclick,
      onnodedrag,
      onnodedragstart,
      onnodedragstop,
      onnodepointerenter,
      onnodepointerleave,
      onnodepointermove,
      onnodecontextmenu
    } = $$props;
    let data = derived(() => fallback(node.data, () => ({}), true)), selected = derived(() => fallback(node.selected, false)), _draggable = derived(() => node.draggable), _selectable = derived(() => node.selectable), deletable = derived(() => fallback(node.deletable, true)), _connectable = derived(() => node.connectable), _focusable = derived(() => node.focusable), hidden = derived(() => fallback(node.hidden, false)), dragging = derived(() => fallback(node.dragging, false)), style = derived(() => fallback(node.style, "")), className = derived(() => node.class), type = derived(() => fallback(node.type, "default")), parentId = derived(() => node.parentId), sourcePosition = derived(() => node.sourcePosition), targetPosition = derived(() => node.targetPosition), measuredWidth = derived(() => fallback(node.measured, () => ({ width: 0, height: 0 }), true).width), measuredHeight = derived(() => fallback(node.measured, () => ({ width: 0, height: 0 }), true).height), initialWidth = derived(() => node.initialWidth), initialHeight = derived(() => node.initialHeight), width = derived(() => node.width), height = derived(() => node.height), dragHandle = derived(() => node.dragHandle), zIndex = derived(() => fallback(node.internals.z, 0)), positionX = derived(() => node.internals.positionAbsolute.x), positionY = derived(() => node.internals.positionAbsolute.y);
    let { id } = node;
    let draggable = derived(() => _draggable() ?? store.nodesDraggable);
    let selectable = derived(() => _selectable() ?? store.elementsSelectable);
    let connectable = derived(() => _connectable() ?? store.nodesConnectable);
    let hasDimensions = derived(() => nodeHasDimensions(node));
    let focusable = derived(() => _focusable() ?? store.nodesFocusable);
    function isInParentLookup(id2) {
      return store.parentLookup.has(id2);
    }
    let isParent = derived(() => isInParentLookup(id));
    type();
    sourcePosition();
    targetPosition();
    let NodeComponent = derived(() => store.nodeTypes[type()] ?? DefaultNode);
    let connectableContext = {
      get value() {
        return connectable();
      }
    };
    setNodeIdContext(id);
    setNodeConnectableContext(connectableContext);
    if (process.env.NODE_ENV === "development") ;
    let nodeStyle = derived(() => {
      const w = measuredWidth() === void 0 ? width() ?? initialWidth() : width();
      const h = measuredHeight() === void 0 ? height() ?? initialHeight() : height();
      if (w === void 0 && h === void 0 && style() === void 0) {
        return void 0;
      }
      return `${style()};${w ? `width:${toPxString(w)};` : ""}${h ? `height:${toPxString(h)};` : ""}`;
    });
    onDestroy(() => {
    });
    if (!hidden()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div${attributes(
        {
          "data-id": id,
          class: clsx([
            "svelte-flow__node",
            `svelte-flow__node-${type()}`,
            className()
          ]),
          style: nodeStyle(),
          tabindex: focusable() ? 0 : void 0,
          role: node.ariaRole ?? (focusable() ? "group" : void 0),
          "aria-label": node.ariaLabel,
          "aria-roledescription": "node",
          "aria-describedby": store.disableKeyboardA11y ? void 0 : `${ARIA_NODE_DESC_KEY}-${store.flowId}`,
          ...node.domAttributes
        },
        void 0,
        {
          dragging,
          selected,
          draggable,
          connectable,
          selectable,
          nopan: draggable(),
          parent: isParent()
        },
        {
          "z-index": zIndex(),
          transform: `translate(${stringify(positionX())}px, ${stringify(positionY())}px)`,
          visibility: hasDimensions() ? "visible" : "hidden"
        }
      )}>`);
      if (NodeComponent()) {
        $$renderer2.push("<!--[-->");
        NodeComponent()($$renderer2, {
          data: data(),
          id,
          selected: selected(),
          selectable: selectable(),
          deletable: deletable(),
          sourcePosition: sourcePosition(),
          targetPosition: targetPosition(),
          zIndex: zIndex(),
          dragging: dragging(),
          draggable: draggable(),
          dragHandle: dragHandle(),
          parentId: parentId(),
          type: type(),
          isConnectable: connectable(),
          positionAbsoluteX: positionX(),
          positionAbsoluteY: positionY(),
          width: width(),
          height: height()
        });
        $$renderer2.push("<!--]-->");
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push("<!--]-->");
      }
      $$renderer2.push(`</div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { store });
  });
}
function NodeRenderer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      store = void 0,
      nodeClickDistance,
      onnodeclick,
      onnodecontextmenu,
      onnodepointerenter,
      onnodepointermove,
      onnodepointerleave,
      onnodedrag,
      onnodedragstart,
      onnodedragstop
    } = $$props;
    const resizeObserver = typeof ResizeObserver === "undefined" ? null : new ResizeObserver((entries) => {
      const updates = /* @__PURE__ */ new Map();
      entries.forEach((entry) => {
        const id = entry.target.getAttribute("data-id");
        updates.set(id, { id, nodeElement: entry.target, force: true });
      });
      store.updateNodeInternals(updates);
    });
    onDestroy(() => {
      resizeObserver?.disconnect();
    });
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="svelte-flow__nodes"><!--[-->`);
      const each_array = ensure_array_like(store.visible.nodes.values());
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let node = each_array[$$index];
        NodeWrapper($$renderer3, {
          node,
          resizeObserver,
          nodeClickDistance,
          onnodeclick,
          onnodepointerenter,
          onnodepointermove,
          onnodepointerleave,
          onnodedrag,
          onnodedragstart,
          onnodedragstop,
          onnodecontextmenu,
          get store() {
            return store;
          },
          set store($$value) {
            store = $$value;
            $$settled = false;
          }
        });
      }
      $$renderer3.push(`<!--]--></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { store });
  });
}
function EdgeWrapper($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const {
      edge,
      store = void 0,
      onedgeclick,
      onedgecontextmenu,
      onedgepointerenter,
      onedgepointerleave
    } = $$props;
    let id = derived(() => edge.id), source = derived(() => edge.source), target = derived(() => edge.target), sourceX = derived(() => edge.sourceX), sourceY = derived(() => edge.sourceY), targetX = derived(() => edge.targetX), targetY = derived(() => edge.targetY), sourcePosition = derived(() => edge.sourcePosition), targetPosition = derived(() => edge.targetPosition), animated = derived(() => fallback(edge.animated, false)), selected = derived(() => fallback(edge.selected, false)), label = derived(() => edge.label), labelStyle = derived(() => edge.labelStyle), data = derived(() => fallback(edge.data, () => ({}), true)), style = derived(() => edge.style), interactionWidth = derived(() => edge.interactionWidth), type = derived(() => fallback(edge.type, "default")), sourceHandle = derived(() => edge.sourceHandle), targetHandle = derived(() => edge.targetHandle), markerStart = derived(() => edge.markerStart), markerEnd = derived(() => edge.markerEnd), _selectable = derived(() => edge.selectable), _focusable = derived(() => edge.focusable), deletable = derived(() => fallback(edge.deletable, true)), hidden = derived(() => edge.hidden), zIndex = derived(() => edge.zIndex), className = derived(() => edge.class), ariaLabel = derived(() => edge.ariaLabel);
    setEdgeIdContext(id());
    let selectable = derived(() => _selectable() ?? store.elementsSelectable);
    let focusable = derived(() => _focusable() ?? store.edgesFocusable);
    let EdgeComponent = derived(() => store.edgeTypes[type()] ?? BezierEdge);
    let markerStartUrl = derived(() => markerStart() ? `url('#${getMarkerId(markerStart(), store.flowId)}')` : void 0);
    let markerEndUrl = derived(() => markerEnd() ? `url('#${getMarkerId(markerEnd(), store.flowId)}')` : void 0);
    if (!hidden()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<svg class="svelte-flow__edge-wrapper"${attr_style("", { "z-index": zIndex() })}><g${attributes(
        {
          class: clsx(["svelte-flow__edge", className()]),
          "data-id": id(),
          "aria-label": ariaLabel() === null ? void 0 : ariaLabel() ? ariaLabel() : `Edge from ${source()} to ${target()}`,
          "aria-describedby": focusable() ? `${ARIA_EDGE_DESC_KEY}-${store.flowId}` : void 0,
          role: edge.ariaRole ?? (focusable() ? "group" : "img"),
          "aria-roledescription": "edge",
          tabindex: focusable() ? 0 : void 0,
          ...edge.domAttributes
        },
        void 0,
        { animated, selected, selectable },
        void 0,
        3
      )}>`);
      if (EdgeComponent()) {
        $$renderer2.push("<!--[-->");
        EdgeComponent()($$renderer2, {
          id: id(),
          source: source(),
          target: target(),
          sourceX: sourceX(),
          sourceY: sourceY(),
          targetX: targetX(),
          targetY: targetY(),
          sourcePosition: sourcePosition(),
          targetPosition: targetPosition(),
          animated: animated(),
          selected: selected(),
          label: label(),
          labelStyle: labelStyle(),
          data: data(),
          style: style(),
          interactionWidth: interactionWidth(),
          selectable: selectable(),
          deletable: deletable(),
          type: type(),
          sourceHandleId: sourceHandle(),
          targetHandleId: targetHandle(),
          markerStart: markerStartUrl(),
          markerEnd: markerEndUrl()
        });
        $$renderer2.push("<!--]-->");
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push("<!--]-->");
      }
      $$renderer2.push(`</g></svg>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { store });
  });
}
function MarkerDefinition($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const store = useStore();
    $$renderer2.push(`<defs><!--[-->`);
    const each_array = ensure_array_like(store.markers);
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let marker = each_array[$$index];
      Marker($$renderer2, spread_props([marker]));
    }
    $$renderer2.push(`<!--]--></defs>`);
  });
}
function Marker($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      id,
      type,
      width = 12.5,
      height = 12.5,
      markerUnits = "strokeWidth",
      orient = "auto-start-reverse",
      color = "none",
      strokeWidth
    } = $$props;
    $$renderer2.push(`<marker class="svelte-flow__arrowhead"${attr("id", id)}${attr("markerWidth", `${width}`)}${attr("markerHeight", `${height}`)} viewBox="-10 -10 20 20"${attr("markerUnits", markerUnits)}${attr("orient", orient)} refX="0" refY="0">`);
    if (type === MarkerType.Arrow) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<polyline class="arrow" fill="none" stroke-linecap="round" stroke-linejoin="round"${attr("stroke-width", strokeWidth)} points="-5,-4 0,0 -5,4"${attr_style("", { stroke: color })}></polyline>`);
    } else if (type === MarkerType.ArrowClosed) {
      $$renderer2.push("<!--[1-->");
      $$renderer2.push(`<polyline class="arrowclosed" stroke-linecap="round" stroke-linejoin="round"${attr("stroke-width", strokeWidth)} points="-5,-4 0,0 -5,4 -5,-4"${attr_style("", { stroke: color, fill: color })}></polyline>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></marker>`);
  });
}
function EdgeRenderer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      store = void 0,
      onedgeclick,
      onedgecontextmenu,
      onedgepointerenter,
      onedgepointerleave
    } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="svelte-flow__edges"><svg class="svelte-flow__marker">`);
      MarkerDefinition($$renderer3);
      $$renderer3.push(`<!----></svg> <!--[-->`);
      const each_array = ensure_array_like(store.visible.edges.values());
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let edge = each_array[$$index];
        EdgeWrapper($$renderer3, {
          edge,
          onedgeclick,
          onedgecontextmenu,
          onedgepointerenter,
          onedgepointerleave,
          get store() {
            return store;
          },
          set store($$value) {
            store = $$value;
            $$settled = false;
          }
        });
      }
      $$renderer3.push(`<!--]--></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { store });
  });
}
function Selection($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { x = 0, y = 0, width = 0, height = 0, isVisible = true } = $$props;
    if (isVisible) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="svelte-flow__selection svelte-1h4lex9"${attr_style("", {
        width: typeof width === "string" ? width : toPxString(width),
        height: typeof height === "string" ? height : toPxString(height),
        transform: `translate(${x}px, ${y}px)`
      })}></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function NodeSelection($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      store = void 0,
      onnodedrag,
      onnodedragstart,
      onnodedragstop,
      onselectionclick,
      onselectioncontextmenu
    } = $$props;
    let bounds = derived(() => {
      if (store.selectionRectMode === "nodes") {
        store.nodes;
        const nodeBounds = getInternalNodesBounds(store.nodeLookup, { filter: (node) => !!node.selected });
        if (nodeBounds.width > 0 && nodeBounds.height > 0) {
          return nodeBounds;
        }
      }
      return null;
    });
    if (store.selectionRectMode === "nodes" && bounds() && isNumeric(bounds().x) && isNumeric(bounds().y)) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div${attr_class(clsx(["svelte-flow__selection-wrapper", store.noPanClass]), "svelte-1i4d9pt")}${attr("role", store.disableKeyboardA11y ? void 0 : "button")}${attr("tabindex", store.disableKeyboardA11y ? void 0 : -1)}${attr_style("", {
        width: toPxString(bounds().width),
        height: toPxString(bounds().height),
        transform: `translate(${stringify(bounds().x)}px, ${stringify(bounds().y)}px)`
      })}>`);
      Selection($$renderer2, { width: "100%", height: "100%", x: 0, y: 0 });
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { store });
  });
}
function useSvelteFlow() {
  const store = derived(useStore);
  const getNodeRect = (node) => {
    const nodeToUse = isNode(node) ? node : store().nodeLookup.get(node.id);
    const position = nodeToUse.parentId ? evaluateAbsolutePosition(nodeToUse.position, nodeToUse.measured, nodeToUse.parentId, store().nodeLookup, store().nodeOrigin) : nodeToUse.position;
    const nodeWithPosition = {
      ...nodeToUse,
      position,
      width: nodeToUse.measured?.width ?? nodeToUse.width,
      height: nodeToUse.measured?.height ?? nodeToUse.height
    };
    return nodeToRect(nodeWithPosition);
  };
  function updateNode(id, nodeUpdate, options = { replace: false }) {
    store().nodes = run(() => store().nodes).map((node) => {
      if (node.id === id) {
        const nextNode = typeof nodeUpdate === "function" ? nodeUpdate(node) : nodeUpdate;
        return options?.replace && isNode(nextNode) ? nextNode : { ...node, ...nextNode };
      }
      return node;
    });
  }
  function updateEdge(id, edgeUpdate, options = { replace: false }) {
    store().edges = run(() => store().edges).map((edge) => {
      if (edge.id === id) {
        const nextEdge = typeof edgeUpdate === "function" ? edgeUpdate(edge) : edgeUpdate;
        return options.replace && isEdge(nextEdge) ? nextEdge : { ...edge, ...nextEdge };
      }
      return edge;
    });
  }
  const getInternalNode = (id) => store().nodeLookup.get(id);
  return {
    zoomIn: store().zoomIn,
    zoomOut: store().zoomOut,
    getInternalNode,
    getNode: (id) => getInternalNode(id)?.internals.userNode,
    getNodes: (ids) => ids === void 0 ? store().nodes : getElements(store().nodeLookup, ids),
    getEdge: (id) => store().edgeLookup.get(id),
    getEdges: (ids) => ids === void 0 ? store().edges : getElements(store().edgeLookup, ids),
    setZoom: async (zoomLevel, options) => {
      const panZoom = store().panZoom;
      return panZoom ? panZoom.scaleTo(zoomLevel, options) : false;
    },
    getZoom: () => store().viewport.zoom,
    setViewport: async (nextViewport, options) => {
      const currentViewport = store().viewport;
      if (!store().panZoom) {
        return false;
      }
      await store().panZoom.setViewport(
        {
          x: nextViewport.x ?? currentViewport.x,
          y: nextViewport.y ?? currentViewport.y,
          zoom: nextViewport.zoom ?? currentViewport.zoom
        },
        options
      );
      return true;
    },
    getViewport: () => snapshot(store().viewport),
    setCenter: async (x, y, options) => store().setCenter(x, y, options),
    fitView: (options) => store().fitView(options),
    fitBounds: async (bounds, options) => {
      if (!store().panZoom) {
        return false;
      }
      const viewport = getViewportForBounds(bounds, store().width, store().height, store().minZoom, store().maxZoom, options?.padding ?? 0.1);
      await store().panZoom.setViewport(viewport, {
        duration: options?.duration,
        ease: options?.ease,
        interpolate: options?.interpolate
      });
      return true;
    },
    /**
     * Partial is defined as "the 2 nodes/areas are intersecting partially".
     * If a is contained in b or b is contained in a, they are both
     * considered fully intersecting.
     */
    getIntersectingNodes: (nodeOrRect, partially = true, nodesToIntersect) => {
      const isRect = isRectObject(nodeOrRect);
      const nodeRect = isRect ? nodeOrRect : getNodeRect(nodeOrRect);
      if (!nodeRect) {
        return [];
      }
      return (nodesToIntersect || store().nodes).filter((n) => {
        const internalNode = store().nodeLookup.get(n.id);
        if (!internalNode || !isRect && n.id === nodeOrRect.id) {
          return false;
        }
        const currNodeRect = nodeToRect(internalNode);
        const overlappingArea = getOverlappingArea(currNodeRect, nodeRect);
        const partiallyVisible = partially && overlappingArea > 0;
        return partiallyVisible || overlappingArea >= currNodeRect.width * currNodeRect.height || overlappingArea >= nodeRect.width * nodeRect.height;
      });
    },
    isNodeIntersecting: (nodeOrRect, area, partially = true) => {
      const isRect = isRectObject(nodeOrRect);
      const nodeRect = isRect ? nodeOrRect : getNodeRect(nodeOrRect);
      if (!nodeRect) {
        return false;
      }
      const overlappingArea = getOverlappingArea(nodeRect, area);
      const partiallyVisible = partially && overlappingArea > 0;
      return partiallyVisible || overlappingArea >= area.width * area.height || overlappingArea >= nodeRect.width * nodeRect.height;
    },
    deleteElements: async ({ nodes: nodesToRemove = [], edges: edgesToRemove = [] }) => {
      const { nodes: matchingNodes, edges: matchingEdges } = await getElementsToRemove({
        nodesToRemove,
        edgesToRemove,
        nodes: store().nodes,
        edges: store().edges,
        onBeforeDelete: store().onbeforedelete
      });
      if (matchingNodes) {
        store().nodes = run(() => store().nodes).filter((node) => !matchingNodes.some(({ id }) => id === node.id));
      }
      if (matchingEdges) {
        store().edges = run(() => store().edges).filter((edge) => !matchingEdges.some(({ id }) => id === edge.id));
      }
      if (matchingNodes.length > 0 || matchingEdges.length > 0) {
        store().ondelete?.({ nodes: matchingNodes, edges: matchingEdges });
      }
      return { deletedNodes: matchingNodes, deletedEdges: matchingEdges };
    },
    screenToFlowPosition: (position, options = { snapToGrid: true }) => {
      if (!store().domNode) {
        return position;
      }
      const _snapGrid = options.snapToGrid ? store().snapGrid : false;
      const { x, y, zoom } = store().viewport;
      const { x: domX, y: domY } = store().domNode.getBoundingClientRect();
      const correctedPosition = { x: position.x - domX, y: position.y - domY };
      return pointToRendererPoint(correctedPosition, [x, y, zoom], _snapGrid !== null, _snapGrid || [1, 1]);
    },
    /**
     *
     * @param position
     * @returns
     */
    flowToScreenPosition: (position) => {
      if (!store().domNode) {
        return position;
      }
      const { x, y, zoom } = store().viewport;
      const { x: domX, y: domY } = store().domNode.getBoundingClientRect();
      const rendererPosition = rendererPointToPoint(position, [x, y, zoom]);
      return { x: rendererPosition.x + domX, y: rendererPosition.y + domY };
    },
    toObject: () => {
      return structuredClone({
        nodes: [...store().nodes],
        edges: [...store().edges],
        viewport: { ...store().viewport }
      });
    },
    updateNode,
    updateNodeData: (id, dataUpdate, options) => {
      const node = store().nodeLookup.get(id)?.internals.userNode;
      if (!node) {
        return;
      }
      const nextData = typeof dataUpdate === "function" ? dataUpdate(node) : dataUpdate;
      updateNode(id, (node2) => ({
        ...node2,
        data: options?.replace ? nextData : { ...node2.data, ...nextData }
      }));
    },
    updateEdge,
    getNodesBounds: (nodes) => {
      return getNodesBounds(nodes, {
        nodeLookup: store().nodeLookup,
        nodeOrigin: store().nodeOrigin
      });
    },
    getHandleConnections: ({ type, id, nodeId }) => Array.from(store().connectionLookup.get(`${nodeId}-${type}-${id ?? null}`)?.values() ?? [])
  };
}
function getElements(lookup, ids) {
  const result = [];
  for (const id of ids) {
    const item = lookup.get(id);
    if (item) {
      const element = "internals" in item ? item.internals?.userNode : item;
      result.push(element);
    }
  }
  return result;
}
function KeyHandler($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      store = void 0,
      selectionKey = "Shift",
      multiSelectionKey = isMacOs() ? "Meta" : "Control",
      deleteKey = "Backspace",
      panActivationKey = " ",
      zoomActivationKey = isMacOs() ? "Meta" : "Control"
    } = $$props;
    useSvelteFlow();
    bind_props($$props, { store });
  });
}
function ConnectionLine($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { store = void 0, type, containerStyle, style, LineComponent } = $$props;
    let path = derived(() => {
      if (!store.connection.inProgress) {
        return "";
      }
      const pathParams = {
        sourceX: store.connection.from.x,
        sourceY: store.connection.from.y,
        sourcePosition: store.connection.fromPosition,
        targetX: store.connection.to.x,
        targetY: store.connection.to.y,
        targetPosition: store.connection.toPosition
      };
      switch (type) {
        case ConnectionLineType.Bezier: {
          const [path2] = getBezierPath(pathParams);
          return path2;
        }
        case ConnectionLineType.Straight: {
          const [path2] = getStraightPath(pathParams);
          return path2;
        }
        case ConnectionLineType.Step:
        case ConnectionLineType.SmoothStep: {
          const [path2] = getSmoothStepPath({
            ...pathParams,
            borderRadius: type === ConnectionLineType.Step ? 0 : void 0
          });
          return path2;
        }
      }
    });
    if (store.connection.inProgress) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<svg${attr("width", store.width)}${attr("height", store.height)} class="svelte-flow__connectionline"${attr_style(containerStyle)}><g${attr_class(clsx([
        "svelte-flow__connection",
        getConnectionStatus(store.connection.isValid)
      ]))}>`);
      if (LineComponent) {
        $$renderer2.push("<!--[0-->");
        if (LineComponent) {
          $$renderer2.push("<!--[-->");
          LineComponent($$renderer2, {});
          $$renderer2.push("<!--]-->");
        } else {
          $$renderer2.push("<!--[!-->");
          $$renderer2.push("<!--]-->");
        }
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<path${attr("d", path())}${attr_style(style)} fill="none" class="svelte-flow__connection-path"></path>`);
      }
      $$renderer2.push(`<!--]--></g></svg>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { store });
  });
}
function Panel($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      position = "top-right",
      style,
      class: className,
      children,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    let positionClasses = derived(() => `${position}`.split("-"));
    $$renderer2.push(`<div${attributes({
      class: clsx(["svelte-flow__panel", className, ...positionClasses()]),
      style,
      ...rest
    })}>`);
    children?.($$renderer2);
    $$renderer2.push(`<!----></div>`);
  });
}
function Attribution($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { proOptions, position = "bottom-right" } = $$props;
    if (!proOptions?.hideAttribution) {
      $$renderer2.push("<!--[0-->");
      Panel($$renderer2, {
        position,
        class: "svelte-flow__attribution",
        "data-message": "Feel free to remove the attribution or check out how you could support us: https://svelteflow.dev/support-us",
        children: ($$renderer3) => {
          $$renderer3.push(`<a href="https://svelteflow.dev" target="_blank" rel="noopener noreferrer" aria-label="Svelte Flow attribution">Svelte Flow</a>`);
        },
        $$slots: { default: true }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function Wrapper($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      width,
      height,
      colorMode,
      domNode = void 0,
      clientWidth = void 0,
      clientHeight = void 0,
      children,
      rest
    } = $$props;
    let className = derived(() => rest.class), divAttributes = derived(() => exclude_from_object(rest, [
      "id",
      "class",
      "nodeTypes",
      "edgeTypes",
      "colorMode",
      "isValidConnection",
      "onmove",
      "onmovestart",
      "onmoveend",
      "onflowerror",
      "ondelete",
      "onbeforedelete",
      "onbeforeconnect",
      "onconnect",
      "onconnectstart",
      "onconnectend",
      "onbeforereconnect",
      "onreconnect",
      "onreconnectstart",
      "onreconnectend",
      "onclickconnectstart",
      "onclickconnectend",
      "oninit",
      "onselectionchange",
      "onselectiondragstart",
      "onselectiondrag",
      "onselectiondragstop",
      "onselectionstart",
      "onselectionend",
      "clickConnect",
      "fitView",
      "fitViewOptions",
      "nodeOrigin",
      "nodeDragThreshold",
      "connectionDragThreshold",
      "minZoom",
      "maxZoom",
      "initialViewport",
      "connectionRadius",
      "connectionMode",
      "selectionMode",
      "selectNodesOnDrag",
      "snapGrid",
      "defaultMarkerColor",
      "translateExtent",
      "nodeExtent",
      "onlyRenderVisibleElements",
      "autoPanOnConnect",
      "autoPanOnNodeDrag",
      "colorModeSSR",
      "defaultEdgeOptions",
      "elevateNodesOnSelect",
      "elevateEdgesOnSelect",
      "nodesDraggable",
      "autoPanOnNodeFocus",
      "nodesConnectable",
      "elementsSelectable",
      "nodesFocusable",
      "edgesFocusable",
      "disableKeyboardA11y",
      "noDragClass",
      "noPanClass",
      "noWheelClass",
      "ariaLabelConfig",
      "autoPanSpeed",
      "panOnScrollSpeed",
      "zIndexMode",
      "autoPanOnSelection"
    ]));
    $$renderer2.push(`<div${attributes(
      {
        class: clsx([
          "svelte-flow",
          "svelte-flow__container",
          colorMode,
          className()
        ]),
        "data-testid": "svelte-flow__wrapper",
        role: "application",
        ...divAttributes()
      },
      "svelte-19yox0q",
      void 0,
      { width: toPxString(width), height: toPxString(height) }
    )}>`);
    children?.($$renderer2);
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { domNode, clientWidth, clientHeight });
  });
}
function SvelteFlow($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      width,
      height,
      proOptions,
      selectionKey,
      deleteKey,
      panActivationKey,
      multiSelectionKey,
      zoomActivationKey,
      paneClickDistance = 1,
      nodeClickDistance = 1,
      onmovestart,
      onmoveend,
      onmove,
      oninit,
      onnodeclick,
      onnodecontextmenu,
      onnodedrag,
      onnodedragstart,
      onnodedragstop,
      onnodepointerenter,
      onnodepointermove,
      onnodepointerleave,
      onselectionclick,
      onselectioncontextmenu,
      onselectionstart,
      onselectionend,
      onedgeclick,
      onedgecontextmenu,
      onedgepointerenter,
      onedgepointerleave,
      onpaneclick,
      onpanecontextmenu,
      panOnScrollMode = PanOnScrollMode.Free,
      preventScrolling = true,
      zoomOnScroll = true,
      zoomOnDoubleClick = true,
      zoomOnPinch = true,
      panOnScroll = false,
      panOnScrollSpeed = 0.5,
      panOnDrag = true,
      selectionOnDrag = false,
      autoPanOnSelection = true,
      connectionLineComponent,
      connectionLineStyle,
      connectionLineContainerStyle,
      connectionLineType = ConnectionLineType.Bezier,
      attributionPosition,
      children,
      nodes = [],
      edges = [],
      viewport = void 0,
      $$slots,
      $$events,
      ...props
    } = $$props;
    let store = createStore({
      props,
      width,
      height,
      get nodes() {
        return nodes;
      },
      set nodes(newNodes) {
        nodes = newNodes;
      },
      get edges() {
        return edges;
      },
      set edges(newEdges) {
        edges = newEdges;
      },
      get viewport() {
        return viewport;
      },
      set viewport(newViewport) {
        viewport = newViewport;
      }
    });
    const providerContext = getContext(key);
    if (providerContext && providerContext.setStore) {
      providerContext.setStore(store);
    }
    setContext(key, {
      provider: false,
      getStore() {
        return store;
      }
    });
    onDestroy(() => {
      store.reset();
    });
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Wrapper($$renderer3, {
        colorMode: store.colorMode,
        width,
        height,
        rest: props,
        get domNode() {
          return store.domNode;
        },
        set domNode($$value) {
          store.domNode = $$value;
          $$settled = false;
        },
        get clientWidth() {
          return store.width;
        },
        set clientWidth($$value) {
          store.width = $$value;
          $$settled = false;
        },
        get clientHeight() {
          return store.height;
        },
        set clientHeight($$value) {
          store.height = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          KeyHandler($$renderer4, {
            selectionKey,
            deleteKey,
            panActivationKey,
            multiSelectionKey,
            zoomActivationKey,
            get store() {
              return store;
            },
            set store($$value) {
              store = $$value;
              $$settled = false;
            }
          });
          $$renderer4.push(`<!----> `);
          Zoom($$renderer4, {
            panOnScrollMode,
            preventScrolling,
            zoomOnScroll,
            zoomOnDoubleClick,
            zoomOnPinch,
            panOnScroll,
            panOnScrollSpeed,
            panOnDrag,
            paneClickDistance,
            selectionOnDrag,
            onmovestart,
            onmove,
            onmoveend,
            oninit,
            get store() {
              return store;
            },
            set store($$value) {
              store = $$value;
              $$settled = false;
            },
            children: ($$renderer5) => {
              Pane($$renderer5, {
                onpaneclick,
                onpanecontextmenu,
                onselectionstart,
                onselectionend,
                panOnDrag,
                paneClickDistance,
                selectionOnDrag,
                autoPanOnSelection,
                get store() {
                  return store;
                },
                set store($$value) {
                  store = $$value;
                  $$settled = false;
                },
                children: ($$renderer6) => {
                  Viewport($$renderer6, {
                    get store() {
                      return store;
                    },
                    set store($$value) {
                      store = $$value;
                      $$settled = false;
                    },
                    children: ($$renderer7) => {
                      $$renderer7.push(`<div class="svelte-flow__viewport-back svelte-flow__container"></div> `);
                      EdgeRenderer($$renderer7, {
                        onedgeclick,
                        onedgecontextmenu,
                        onedgepointerenter,
                        onedgepointerleave,
                        get store() {
                          return store;
                        },
                        set store($$value) {
                          store = $$value;
                          $$settled = false;
                        }
                      });
                      $$renderer7.push(`<!----> <div class="svelte-flow__edge-labels svelte-flow__container"></div> `);
                      ConnectionLine($$renderer7, {
                        type: connectionLineType,
                        LineComponent: connectionLineComponent,
                        containerStyle: connectionLineContainerStyle,
                        style: connectionLineStyle,
                        get store() {
                          return store;
                        },
                        set store($$value) {
                          store = $$value;
                          $$settled = false;
                        }
                      });
                      $$renderer7.push(`<!----> `);
                      NodeRenderer($$renderer7, {
                        nodeClickDistance,
                        onnodeclick,
                        onnodecontextmenu,
                        onnodepointerenter,
                        onnodepointermove,
                        onnodepointerleave,
                        onnodedrag,
                        onnodedragstart,
                        onnodedragstop,
                        get store() {
                          return store;
                        },
                        set store($$value) {
                          store = $$value;
                          $$settled = false;
                        }
                      });
                      $$renderer7.push(`<!----> `);
                      NodeSelection($$renderer7, {
                        onselectionclick,
                        onselectioncontextmenu,
                        onnodedrag,
                        onnodedragstart,
                        onnodedragstop,
                        get store() {
                          return store;
                        },
                        set store($$value) {
                          store = $$value;
                          $$settled = false;
                        }
                      });
                      $$renderer7.push(`<!----> <div class="svelte-flow__viewport-front svelte-flow__container"></div>`);
                    },
                    $$slots: { default: true }
                  });
                  $$renderer6.push(`<!----> `);
                  Selection($$renderer6, {
                    isVisible: !!(store.selectionRect && store.selectionRectMode === "user"),
                    width: store.selectionRect?.width,
                    height: store.selectionRect?.height,
                    x: store.selectionRect?.x,
                    y: store.selectionRect?.y
                  });
                  $$renderer6.push(`<!---->`);
                },
                $$slots: { default: true }
              });
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          Attribution($$renderer4, { proOptions, position: attributionPosition });
          $$renderer4.push(`<!----> `);
          A11yDescriptions($$renderer4, { store });
          $$renderer4.push(`<!----> `);
          children?.($$renderer4);
          $$renderer4.push(`<!---->`);
        },
        $$slots: { default: true }
      });
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { nodes, edges, viewport });
  });
}
var BackgroundVariant;
(function(BackgroundVariant2) {
  BackgroundVariant2["Lines"] = "lines";
  BackgroundVariant2["Dots"] = "dots";
  BackgroundVariant2["Cross"] = "cross";
})(BackgroundVariant || (BackgroundVariant = {}));
function DotPattern($$renderer, $$props) {
  let { radius, class: className } = $$props;
  $$renderer.push(`<circle${attr("cx", radius)}${attr("cy", radius)}${attr("r", radius)}${attr_class(clsx(["svelte-flow__background-pattern", "dots", className]))}></circle>`);
}
function LinePattern($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { lineWidth, dimensions, variant, class: className } = $$props;
    $$renderer2.push(`<path${attr("stroke-width", lineWidth)}${attr("d", `M${dimensions[0] / 2} 0 V${dimensions[1]} M0 ${dimensions[1] / 2} H${dimensions[0]}`)}${attr_class(clsx(["svelte-flow__background-pattern", variant, className]))}></path>`);
  });
}
const defaultSize = {
  [BackgroundVariant.Dots]: 1,
  [BackgroundVariant.Lines]: 1,
  [BackgroundVariant.Cross]: 6
};
function Background($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      id,
      variant = BackgroundVariant.Dots,
      gap = 20,
      size,
      lineWidth = 1,
      bgColor,
      patternColor,
      patternClass,
      class: className
    } = $$props;
    let store = derived(useStore);
    let isDots = derived(() => variant === BackgroundVariant.Dots);
    let isCross = derived(() => variant === BackgroundVariant.Cross);
    let gapXY = derived(() => Array.isArray(gap) ? gap : [gap, gap]);
    let patternId = derived(() => `background-pattern-${store().flowId}-${id ?? ""}`);
    let scaledGap = derived(() => [
      gapXY()[0] * store().viewport.zoom || 1,
      gapXY()[1] * store().viewport.zoom || 1
    ]);
    let scaledSize = derived(() => (size ?? defaultSize[variant]) * store().viewport.zoom);
    let patternDimensions = derived(() => isCross() ? [scaledSize(), scaledSize()] : scaledGap());
    let patternOffset = derived(() => isDots() ? [scaledSize() / 2, scaledSize() / 2] : [patternDimensions()[0] / 2, patternDimensions()[1] / 2]);
    $$renderer2.push(`<svg${attr_class(clsx([
      "svelte-flow__background",
      "svelte-flow__container",
      className
    ]))} data-testid="svelte-flow__background"${attr_style("", {
      "--xy-background-color-props": bgColor,
      "--xy-background-pattern-color-props": patternColor
    })}><pattern${attr("id", patternId())}${attr("x", store().viewport.x % scaledGap()[0])}${attr("y", store().viewport.y % scaledGap()[1])}${attr("width", scaledGap()[0])}${attr("height", scaledGap()[1])} patternUnits="userSpaceOnUse"${attr("patternTransform", `translate(-${patternOffset()[0]},-${patternOffset()[1]})`)}>`);
    if (isDots()) {
      $$renderer2.push("<!--[0-->");
      DotPattern($$renderer2, { radius: scaledSize() / 2, class: patternClass });
    } else {
      $$renderer2.push("<!--[-1-->");
      LinePattern($$renderer2, {
        dimensions: patternDimensions(),
        variant,
        lineWidth,
        class: patternClass
      });
    }
    $$renderer2.push(`<!--]--></pattern><rect x="0" y="0" width="100%" height="100%"${attr("fill", `url(#${patternId()})`)}></rect></svg>`);
  });
}
export {
  Background as B,
  EdgeLabel as E,
  Handle as H,
  Panel as P,
  SvelteFlow as S,
  BaseEdge as a,
  useSvelteFlow as b,
  createStore as c,
  getEdgeIdContext as g,
  key as k,
  snapshot as s,
  useStore as u
};
