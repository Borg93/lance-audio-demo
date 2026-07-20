import "../../../chunks/async.js";
import "clsx";
import { a as Play, X, P as Plus$1, L as Loader_circle, b as Player_pane, R as Resizable_split } from "../../../chunks/resizable-split.js";
import { c as setContext, a as spread_props, j as bind_props, e as attributes, f as clsx, d as derived, i as attr_class, h as attr, n as attr_style, o as css_props, k as stringify, m as escape_html, l as ensure_array_like, s as sanitize_props, b as slot, p as props_id } from "../../../chunks/renderer.js";
import { c as createStore, k as key, u as useStore, g as getEdgeIdContext, E as EdgeLabel, P as Panel, s as snapshot, H as Handle, a as BaseEdge, b as useSvelteFlow, S as SvelteFlow, B as Background } from "../../../chunks/style.js";
import { D as DEV } from "../../../chunks/false.js";
import { c as activeView, e as search, r as relevanceOf, d as chunkFrameUrl } from "../../../chunks/api.js";
import { C as Context, M as PresenceManager, d as boxWith, w as watch, m as attachRef, Q as getDataOpenClosed, q as createBitsAttrs, R as getDataTransitionAttrs, n as boolToEmptyStrOrUndef, r as createId, u as mergeProps, T as afterTick, x as ENTER, E as END, H as HOME, j as ARROW_LEFT, A as ARROW_UP, U as h, V as k, W as p, h as ARROW_RIGHT, k as ARROW_DOWN, X as l, Y as j, Z as n, o as boolToStr, _ as getFirstNonCommentChild, $ as afterSleep, s as noop, a0 as Focus_scope, a1 as Escape_layer, a2 as Dismissible_layer, a3 as Text_selection_layer, a4 as Scroll_lock, F as hitKey, B as Button, P as Portal } from "../../../chunks/scroll-lock.js";
import ELK from "elkjs/lib/elk.bundled.js";
import * as v from "valibot";
import { I as Icon } from "../../../chunks/Icon.js";
import { XYHandle, getNodeDimensions, nodeHasDimensions, getBoundsOfRects, getInternalNodesBounds, Position, MarkerType, getBezierPath } from "@xyflow/system";
import { M as Map$1 } from "../../../chunks/map.js";
import { t as to_array } from "../../../chunks/render-context.js";
import { o as onDestroy } from "../../../chunks/index-server.js";
import { s as srOnlyStyles } from "../../../chunks/sr-only-styles.js";
function SvelteFlowProvider($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { children } = $$props;
    let store = createStore({ props: {}, nodes: [], edges: [] });
    setContext(key, {
      provider: true,
      getStore() {
        return store;
      },
      setStore: (newStore) => {
        store = newStore;
      }
    });
    onDestroy(() => {
      store.reset();
    });
    children?.($$renderer2);
    $$renderer2.push(`<!---->`);
  });
}
function EdgeReconnectAnchor($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      type,
      reconnecting = false,
      position,
      class: className,
      size = 25,
      dragThreshold = 1,
      children,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    const store = useStore();
    const edgeId = getEdgeIdContext("EdgeReconnectAnchor must be used within a Custom Edge component");
    const onPointerDown = (event) => {
      if (event.button !== 0) {
        return;
      }
      const {
        autoPanOnConnect,
        domNode,
        isValidConnection,
        connectionMode,
        connectionRadius,
        onconnectstart,
        onconnectend,
        onreconnect,
        onreconnectstart,
        onreconnectend,
        onbeforereconnect,
        cancelConnection,
        nodeLookup,
        flowId,
        panBy,
        updateConnection,
        edgeLookup
      } = store;
      let edge = edgeLookup.get(edgeId);
      const _onConnectStart = (evt, params) => {
        reconnecting = true;
        onreconnectstart?.(event, edge, type);
        onconnectstart?.(evt, params);
      };
      const opposite = type === "target" ? {
        nodeId: edge.source,
        handleId: edge.sourceHandle ?? null,
        type: "source"
      } : {
        nodeId: edge.target,
        handleId: edge.targetHandle ?? null,
        type: "target"
      };
      XYHandle.onPointerDown(event, {
        autoPanOnConnect,
        connectionMode,
        connectionRadius,
        domNode,
        handleId: opposite.handleId,
        nodeId: opposite.nodeId,
        nodeLookup,
        isTarget: opposite.type === "target",
        edgeUpdaterType: opposite.type,
        lib: "svelte",
        flowId,
        cancelConnection,
        panBy,
        isValidConnection: (...args) => store.isValidConnection?.(...args) ?? true,
        onConnectStart: _onConnectStart,
        onConnectEnd: (...args) => store.onconnectend?.(...args),
        onConnect: (connection) => {
          const reconnectedEdge = { ...edge, ...connection };
          const newEdge = onbeforereconnect ? onbeforereconnect(reconnectedEdge, edge) : reconnectedEdge;
          if (!newEdge) {
            return;
          }
          store.edges = store.edges.map((e) => e.id === edge.id ? newEdge : e);
          onreconnect?.(edge, connection);
        },
        onReconnectEnd: (event2, connectionState) => {
          reconnecting = false;
          onreconnectend?.(event2, edge, opposite.type, connectionState);
        },
        updateConnection,
        getTransform: () => [store.viewport.x, store.viewport.y, store.viewport.zoom],
        getFromHandle: () => store.connection.fromHandle,
        dragThreshold: dragThreshold ?? store.connectionDragThreshold,
        handleDomNode: event.currentTarget
      });
    };
    EdgeLabel($$renderer2, spread_props([
      {
        x: position?.x,
        y: position?.y,
        width: size,
        height: size,
        class: [
          "svelte-flow__edgeupdater",
          `svelte-flow__edgeupdater-${type}`,
          store.noPanClass,
          className
        ],
        onpointerdown: onPointerDown,
        transparent: true
      },
      rest,
      {
        children: ($$renderer3) => {
          if (!reconnecting && children) {
            $$renderer3.push("<!--[0-->");
            children($$renderer3);
            $$renderer3.push(`<!---->`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]-->`);
        },
        $$slots: { default: true }
      }
    ]));
    bind_props($$props, { reconnecting });
  });
}
function ControlButton($$renderer, $$props) {
  let {
    class: className,
    bgColor,
    bgColorHover,
    color,
    colorHover,
    borderColor,
    onclick,
    children,
    $$slots,
    $$events,
    ...restProps
  } = $$props;
  $$renderer.push(`<button${attributes(
    {
      type: "button",
      class: clsx(["svelte-flow__controls-button", className]),
      ...restProps
    },
    void 0,
    void 0,
    {
      "--xy-controls-button-background-color-props": bgColor,
      "--xy-controls-button-background-color-hover-props": bgColorHover,
      "--xy-controls-button-color-props": color,
      "--xy-controls-button-color-hover-props": colorHover,
      "--xy-controls-button-border-color-props": borderColor
    }
  )}>`);
  children?.($$renderer);
  $$renderer.push(`<!----></button>`);
}
function Plus($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M32 18.133H18.133V32h-4.266V18.133H0v-4.266h13.867V0h4.266v13.867H32z"></path></svg>`);
}
function Minus($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 5"><path d="M0 0h32v4.2H0z"></path></svg>`);
}
function Fit($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 30"><path d="M3.692 4.63c0-.53.4-.938.939-.938h5.215V0H4.708C2.13 0 0 2.054 0 4.63v5.216h3.692V4.631zM27.354 0h-5.2v3.692h5.17c.53 0 .984.4.984.939v5.215H32V4.631A4.624 4.624 0 0027.354 0zm.954 24.83c0 .532-.4.94-.939.94h-5.215v3.768h5.215c2.577 0 4.631-2.13 4.631-4.707v-5.139h-3.692v5.139zm-23.677.94c-.531 0-.939-.4-.939-.94v-5.138H0v5.139c0 2.577 2.13 4.707 4.708 4.707h5.138V25.77H4.631z"></path></svg>`);
}
function Lock($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 25 32"><path d="M21.333 10.667H19.81V7.619C19.81 3.429 16.38 0 12.19 0 8 0 4.571 3.429 4.571 7.619v3.048H3.048A3.056 3.056 0 000 13.714v15.238A3.056 3.056 0 003.048 32h18.285a3.056 3.056 0 003.048-3.048V13.714a3.056 3.056 0 00-3.048-3.047zM12.19 24.533a3.056 3.056 0 01-3.047-3.047 3.056 3.056 0 013.047-3.048 3.056 3.056 0 013.048 3.048 3.056 3.056 0 01-3.048 3.047zm4.724-13.866H7.467V7.619c0-2.59 2.133-4.724 4.723-4.724 2.591 0 4.724 2.133 4.724 4.724v3.048z"></path></svg>`);
}
function Unlock($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 25 32"><path d="M21.333 10.667H19.81V7.619C19.81 3.429 16.38 0 12.19 0c-4.114 1.828-1.37 2.133.305 2.438 1.676.305 4.42 2.59 4.42 5.181v3.048H3.047A3.056 3.056 0 000 13.714v15.238A3.056 3.056 0 003.048 32h18.285a3.056 3.056 0 003.048-3.048V13.714a3.056 3.056 0 00-3.048-3.047zM12.19 24.533a3.056 3.056 0 01-3.047-3.047 3.056 3.056 0 013.047-3.048 3.056 3.056 0 013.048 3.048 3.056 3.056 0 01-3.048 3.047z"></path></svg>`);
}
function Controls($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      position = "bottom-left",
      orientation = "vertical",
      showZoom = true,
      showFitView = true,
      showLock = true,
      style,
      class: className,
      buttonBgColor,
      buttonBgColorHover,
      buttonColor,
      buttonColorHover,
      buttonBorderColor,
      fitViewOptions,
      children,
      before,
      after,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    let store = derived(useStore);
    const buttonProps = derived(() => ({
      bgColor: buttonBgColor,
      bgColorHover: buttonBgColorHover,
      color: buttonColor,
      colorHover: buttonColorHover,
      borderColor: buttonBorderColor
    }));
    let isInteractive = derived(() => store().nodesDraggable || store().nodesConnectable || store().elementsSelectable);
    let minZoomReached = derived(() => store().viewport.zoom <= store().minZoom);
    let maxZoomReached = derived(() => store().viewport.zoom >= store().maxZoom);
    let ariaLabelConfig = derived(() => store().ariaLabelConfig);
    let orientationClass = derived(() => orientation === "horizontal" ? "horizontal" : "vertical");
    const onZoomInHandler = () => {
      store().zoomIn();
    };
    const onZoomOutHandler = () => {
      store().zoomOut();
    };
    const onFitViewHandler = () => {
      store().fitView(fitViewOptions);
    };
    const onToggleInteractivity = () => {
      let interactive = !isInteractive();
      store().nodesDraggable = interactive;
      store().nodesConnectable = interactive;
      store().elementsSelectable = interactive;
    };
    Panel($$renderer2, spread_props([
      {
        class: ["svelte-flow__controls", orientationClass(), className],
        position,
        "data-testid": "svelte-flow__controls",
        "aria-label": ariaLabelConfig()["controls.ariaLabel"],
        style
      },
      rest,
      {
        children: ($$renderer3) => {
          if (before) {
            $$renderer3.push("<!--[0-->");
            before($$renderer3);
            $$renderer3.push(`<!---->`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (showZoom) {
            $$renderer3.push("<!--[0-->");
            ControlButton($$renderer3, spread_props([
              {
                onclick: onZoomInHandler,
                class: "svelte-flow__controls-zoomin",
                title: ariaLabelConfig()["controls.zoomIn.ariaLabel"],
                "aria-label": ariaLabelConfig()["controls.zoomIn.ariaLabel"],
                disabled: maxZoomReached()
              },
              buttonProps(),
              {
                children: ($$renderer4) => {
                  Plus($$renderer4);
                },
                $$slots: { default: true }
              }
            ]));
            $$renderer3.push(`<!----> `);
            ControlButton($$renderer3, spread_props([
              {
                onclick: onZoomOutHandler,
                class: "svelte-flow__controls-zoomout",
                title: ariaLabelConfig()["controls.zoomOut.ariaLabel"],
                "aria-label": ariaLabelConfig()["controls.zoomOut.ariaLabel"],
                disabled: minZoomReached()
              },
              buttonProps(),
              {
                children: ($$renderer4) => {
                  Minus($$renderer4);
                },
                $$slots: { default: true }
              }
            ]));
            $$renderer3.push(`<!---->`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (showFitView) {
            $$renderer3.push("<!--[0-->");
            ControlButton($$renderer3, spread_props([
              {
                class: "svelte-flow__controls-fitview",
                onclick: onFitViewHandler,
                title: ariaLabelConfig()["controls.fitView.ariaLabel"],
                "aria-label": ariaLabelConfig()["controls.fitView.ariaLabel"]
              },
              buttonProps(),
              {
                children: ($$renderer4) => {
                  Fit($$renderer4);
                },
                $$slots: { default: true }
              }
            ]));
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (showLock) {
            $$renderer3.push("<!--[0-->");
            ControlButton($$renderer3, spread_props([
              {
                class: "svelte-flow__controls-interactive",
                onclick: onToggleInteractivity,
                title: ariaLabelConfig()["controls.interactive.ariaLabel"],
                "aria-label": ariaLabelConfig()["controls.interactive.ariaLabel"]
              },
              buttonProps(),
              {
                children: ($$renderer4) => {
                  if (isInteractive()) {
                    $$renderer4.push("<!--[0-->");
                    Unlock($$renderer4);
                  } else {
                    $$renderer4.push("<!--[-1-->");
                    Lock($$renderer4);
                  }
                  $$renderer4.push(`<!--]-->`);
                },
                $$slots: { default: true }
              }
            ]));
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (children) {
            $$renderer3.push("<!--[0-->");
            children($$renderer3);
            $$renderer3.push(`<!---->`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (after) {
            $$renderer3.push("<!--[0-->");
            after($$renderer3);
            $$renderer3.push(`<!---->`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]-->`);
        },
        $$slots: { default: true }
      }
    ]));
  });
}
function useInternalNode(id) {
  const $$d = derived(useStore), nodeLookup = derived(() => $$d().nodeLookup), nodes = derived(() => $$d().nodes);
  const node = derived(() => {
    nodes();
    return nodeLookup().get(id);
  });
  return {
    get current() {
      return node();
    }
  };
}
function MinimapNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      id,
      x: xProp,
      y: yProp,
      width: widthProp,
      height: heightProp,
      borderRadius = 5,
      color,
      shapeRendering,
      strokeColor,
      strokeWidth = 2,
      selected,
      class: className,
      nodeComponent
    } = $$props;
    let internalNode = derived(() => useInternalNode(id));
    let $$d = derived(() => {
      if (!internalNode().current) {
        return { width: 0, height: 0, x: 0, y: 0 };
      }
      const { width: width2, height: height2 } = getNodeDimensions(internalNode().current);
      return {
        width: widthProp ?? width2,
        height: heightProp ?? height2,
        x: xProp ?? internalNode().current.internals.positionAbsolute.x,
        y: yProp ?? internalNode().current.internals.positionAbsolute.y
      };
    }), width = derived(() => $$d().width), height = derived(() => $$d().height), x = derived(() => $$d().x), y = derived(() => $$d().y);
    if (nodeComponent) {
      $$renderer2.push("<!--[0-->");
      const CustomComponent = nodeComponent;
      if (CustomComponent) {
        $$renderer2.push("<!--[-->");
        CustomComponent($$renderer2, {
          id,
          x: x(),
          y: y(),
          width: width(),
          height: height(),
          borderRadius,
          class: className,
          color,
          shapeRendering,
          strokeColor,
          strokeWidth,
          selected
        });
        $$renderer2.push("<!--]-->");
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push("<!--]-->");
      }
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<rect${attr_class(clsx(["svelte-flow__minimap-node", className]), void 0, { "selected": selected })}${attr("x", x())}${attr("y", y())}${attr("rx", borderRadius)}${attr("ry", borderRadius)}${attr("width", width())}${attr("height", height())}${attr("shape-rendering", shapeRendering)}${attr_style("", {
        fill: color,
        stroke: strokeColor,
        "stroke-width": strokeWidth
      })}></rect>`);
    }
    $$renderer2.push(`<!--]-->`);
  });
}
const getAttrFunction = (func) => func instanceof Function ? func : () => func;
function Minimap($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      position = "bottom-right",
      ariaLabel,
      nodeStrokeColor = "transparent",
      nodeColor,
      nodeClass = "",
      nodeBorderRadius = 5,
      nodeStrokeWidth = 2,
      nodeComponent,
      bgColor,
      maskColor,
      maskStrokeColor,
      maskStrokeWidth,
      width = 200,
      height = 150,
      pannable = true,
      zoomable = true,
      inversePan,
      zoomStep,
      class: className,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    let store = derived(useStore);
    let ariaLabelConfig = derived(() => store().ariaLabelConfig);
    const shapeRendering = (
      // @ts-expect-error - TS doesn't know about chrome
      typeof window === "undefined" || !!window.chrome ? "crispEdges" : "geometricPrecision"
    );
    let labelledBy = derived(() => `svelte-flow__minimap-desc-${store().flowId}`);
    let viewBB = derived(() => ({
      x: -store().viewport.x / store().viewport.zoom,
      y: -store().viewport.y / store().viewport.zoom,
      width: store().width / store().viewport.zoom,
      height: store().height / store().viewport.zoom
    }));
    let boundingRect = derived(() => getBoundsOfRects(getInternalNodesBounds(store().nodeLookup, { filter: (n2) => !n2.hidden }), viewBB()));
    let scaledWidth = derived(() => boundingRect().width / width);
    let scaledHeight = derived(() => boundingRect().height / height);
    let viewScale = derived(() => Math.max(scaledWidth(), scaledHeight()));
    let viewWidth = derived(() => viewScale() * width);
    let viewHeight = derived(() => viewScale() * height);
    let offset = derived(() => 5 * viewScale());
    let x = derived(() => boundingRect().x - (viewWidth() - boundingRect().width) / 2 - offset());
    let y = derived(() => boundingRect().y - (viewHeight() - boundingRect().height) / 2 - offset());
    let viewboxWidth = derived(() => viewWidth() + offset() * 2);
    let viewboxHeight = derived(() => viewHeight() + offset() * 2);
    css_props($$renderer2, true, { "--xy-minimap-background-color-props": bgColor }, () => {
      Panel($$renderer2, spread_props([
        {
          position,
          class: ["svelte-flow__minimap", className],
          "data-testid": "svelte-flow__minimap"
        },
        rest,
        {
          children: ($$renderer3) => {
            if (store().panZoom) {
              $$renderer3.push("<!--[0-->");
              $$renderer3.push(`<svg${attr("width", width)}${attr("height", height)}${attr("viewBox", `${stringify(x())} ${stringify(y())} ${stringify(viewboxWidth())} ${stringify(viewboxHeight())}`)} class="svelte-flow__minimap-svg" role="img"${attr("aria-labelledby", labelledBy())}${attr_style("", {
                "--xy-minimap-mask-background-color-props": maskColor,
                "--xy-minimap-mask-stroke-color-props": maskStrokeColor,
                "--xy-minimap-mask-stroke-width-props": maskStrokeWidth ? maskStrokeWidth * viewScale() : void 0
              })}>`);
              if (ariaLabel ?? ariaLabelConfig()["minimap.ariaLabel"]) {
                $$renderer3.push("<!--[0-->");
                $$renderer3.push(`<title${attr("id", labelledBy())}>${escape_html(ariaLabel ?? ariaLabelConfig()["minimap.ariaLabel"])}</title>`);
              } else {
                $$renderer3.push("<!--[-1-->");
              }
              $$renderer3.push(`<!--]--><!--[-->`);
              const each_array = ensure_array_like(store().nodes);
              for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                let userNode = each_array[$$index];
                const node = store().nodeLookup.get(userNode.id);
                if (node && nodeHasDimensions(node) && !node.hidden) {
                  $$renderer3.push("<!--[0-->");
                  MinimapNode($$renderer3, {
                    id: node.id,
                    selected: node.selected,
                    nodeComponent,
                    color: nodeColor === void 0 ? void 0 : getAttrFunction(nodeColor)(userNode),
                    borderRadius: nodeBorderRadius,
                    strokeColor: getAttrFunction(nodeStrokeColor)(userNode),
                    strokeWidth: nodeStrokeWidth,
                    shapeRendering,
                    class: getAttrFunction(nodeClass)(userNode)
                  });
                } else {
                  $$renderer3.push("<!--[-1-->");
                }
                $$renderer3.push(`<!--]-->`);
              }
              $$renderer3.push(`<!--]--><path class="svelte-flow__minimap-mask"${attr("d", `M${stringify(x() - offset())},${stringify(y() - offset())}h${stringify(viewboxWidth() + offset() * 2)}v${stringify(viewboxHeight() + offset() * 2)}h${stringify(-viewboxWidth() - offset() * 2)}z
      M${stringify(viewBB().x)},${stringify(viewBB().y)}h${stringify(viewBB().width)}v${stringify(viewBB().height)}h${stringify(-viewBB().width)}z`)} fill-rule="evenodd" pointer-events="none"></path></svg>`);
            } else {
              $$renderer3.push("<!--[-1-->");
            }
            $$renderer3.push(`<!--]-->`);
          },
          $$slots: { default: true }
        }
      ]));
    });
  });
}
function Arrow_left($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "m12 19-7-7 7-7" }],
    ["path", { "d": "M19 12H5" }]
  ];
  Icon($$renderer, spread_props([
    { name: "arrow-left" },
    $$sanitized_props,
    {
      /**
       * @component @name ArrowLeft
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJtMTIgMTktNy03IDctNyIgLz4KICA8cGF0aCBkPSJNMTkgMTJINSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/arrow-left
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Clipboard_paste($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M15 2H9a1 1 0 0 0-1 1v2c0 .6.4 1 1 1h6c.6 0 1-.4 1-1V3c0-.6-.4-1-1-1Z"
      }
    ],
    [
      "path",
      {
        "d": "M8 4H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2M16 4h2a2 2 0 0 1 2 2v2M11 14h10"
      }
    ],
    ["path", { "d": "m17 10 4 4-4 4" }]
  ];
  Icon($$renderer, spread_props([
    { name: "clipboard-paste" },
    $$sanitized_props,
    {
      /**
       * @component @name ClipboardPaste
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTUgMkg5YTEgMSAwIDAgMC0xIDF2MmMwIC42LjQgMSAxIDFoNmMuNiAwIDEtLjQgMS0xVjNjMC0uNi0uNC0xLTEtMVoiIC8+CiAgPHBhdGggZD0iTTggNEg2YTIgMiAwIDAgMC0yIDJ2MTRhMiAyIDAgMCAwIDIgMmgxMmEyIDIgMCAwIDAgMi0yTTE2IDRoMmEyIDIgMCAwIDEgMiAydjJNMTEgMTRoMTAiIC8+CiAgPHBhdGggZD0ibTE3IDEwIDQgNC00IDQiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/clipboard-paste
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Command$1($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "command" },
    $$sanitized_props,
    {
      /**
       * @component @name Command
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTUgNnYxMmEzIDMgMCAxIDAgMy0zSDZhMyAzIDAgMSAwIDMgM1Y2YTMgMyAwIDEgMC0zIDNoMTJhMyAzIDAgMSAwLTMtMyIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/command
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Copy($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "rect",
      {
        "width": "14",
        "height": "14",
        "x": "8",
        "y": "8",
        "rx": "2",
        "ry": "2"
      }
    ],
    [
      "path",
      {
        "d": "M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "copy" },
    $$sanitized_props,
    {
      /**
       * @component @name Copy
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cmVjdCB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHg9IjgiIHk9IjgiIHJ4PSIyIiByeT0iMiIgLz4KICA8cGF0aCBkPSJNNCAxNmMtMS4xIDAtMi0uOS0yLTJWNGMwLTEuMS45LTIgMi0yaDEwYzEuMSAwIDIgLjkgMiAyIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/copy
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Download($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" }],
    ["polyline", { "points": "7 10 12 15 17 10" }],
    ["line", { "x1": "12", "x2": "12", "y1": "15", "y2": "3" }]
  ];
  Icon($$renderer, spread_props([
    { name: "download" },
    $$sanitized_props,
    {
      /**
       * @component @name Download
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMjEgMTV2NGEyIDIgMCAwIDEtMiAySDVhMiAyIDAgMCAxLTItMnYtNCIgLz4KICA8cG9seWxpbmUgcG9pbnRzPSI3IDEwIDEyIDE1IDE3IDEwIiAvPgogIDxsaW5lIHgxPSIxMiIgeDI9IjEyIiB5MT0iMTUiIHkyPSIzIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/download
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Eraser($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"
      }
    ],
    ["path", { "d": "M22 21H7" }],
    ["path", { "d": "m5 11 9 9" }]
  ];
  Icon($$renderer, spread_props([
    { name: "eraser" },
    $$sanitized_props,
    {
      /**
       * @component @name Eraser
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJtNyAyMS00LjMtNC4zYy0xLTEtMS0yLjUgMC0zLjRsOS42LTkuNmMxLTEgMi41LTEgMy40IDBsNS42IDUuNmMxIDEgMSAyLjUgMCAzLjRMMTMgMjEiIC8+CiAgPHBhdGggZD0iTTIyIDIxSDciIC8+CiAgPHBhdGggZD0ibTUgMTEgOSA5IiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/eraser
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Eye_off($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"
      }
    ],
    ["path", { "d": "M14.084 14.158a3 3 0 0 1-4.242-4.242" }],
    [
      "path",
      {
        "d": "M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"
      }
    ],
    ["path", { "d": "m2 2 20 20" }]
  ];
  Icon($$renderer, spread_props([
    { name: "eye-off" },
    $$sanitized_props,
    {
      /**
       * @component @name EyeOff
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTAuNzMzIDUuMDc2YTEwLjc0NCAxMC43NDQgMCAwIDEgMTEuMjA1IDYuNTc1IDEgMSAwIDAgMSAwIC42OTYgMTAuNzQ3IDEwLjc0NyAwIDAgMS0xLjQ0NCAyLjQ5IiAvPgogIDxwYXRoIGQ9Ik0xNC4wODQgMTQuMTU4YTMgMyAwIDAgMS00LjI0Mi00LjI0MiIgLz4KICA8cGF0aCBkPSJNMTcuNDc5IDE3LjQ5OWExMC43NSAxMC43NSAwIDAgMS0xNS40MTctNS4xNTEgMSAxIDAgMCAxIDAtLjY5NiAxMC43NSAxMC43NSAwIDAgMSA0LjQ0Ni01LjE0MyIgLz4KICA8cGF0aCBkPSJtMiAyIDIwIDIwIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/eye-off
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Eye($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"
      }
    ],
    ["circle", { "cx": "12", "cy": "12", "r": "3" }]
  ];
  Icon($$renderer, spread_props([
    { name: "eye" },
    $$sanitized_props,
    {
      /**
       * @component @name Eye
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMi4wNjIgMTIuMzQ4YTEgMSAwIDAgMSAwLS42OTYgMTAuNzUgMTAuNzUgMCAwIDEgMTkuODc2IDAgMSAxIDAgMCAxIDAgLjY5NiAxMC43NSAxMC43NSAwIDAgMS0xOS44NzYgMCIgLz4KICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIzIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/eye
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Grip_vertical($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "9", "cy": "12", "r": "1" }],
    ["circle", { "cx": "9", "cy": "5", "r": "1" }],
    ["circle", { "cx": "9", "cy": "19", "r": "1" }],
    ["circle", { "cx": "15", "cy": "12", "r": "1" }],
    ["circle", { "cx": "15", "cy": "5", "r": "1" }],
    ["circle", { "cx": "15", "cy": "19", "r": "1" }]
  ];
  Icon($$renderer, spread_props([
    { name: "grip-vertical" },
    $$sanitized_props,
    {
      /**
       * @component @name GripVertical
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSI5IiBjeT0iMTIiIHI9IjEiIC8+CiAgPGNpcmNsZSBjeD0iOSIgY3k9IjUiIHI9IjEiIC8+CiAgPGNpcmNsZSBjeD0iOSIgY3k9IjE5IiByPSIxIiAvPgogIDxjaXJjbGUgY3g9IjE1IiBjeT0iMTIiIHI9IjEiIC8+CiAgPGNpcmNsZSBjeD0iMTUiIGN5PSI1IiByPSIxIiAvPgogIDxjaXJjbGUgY3g9IjE1IiBjeT0iMTkiIHI9IjEiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/grip-vertical
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Redo_2($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "m15 14 5-5-5-5" }],
    [
      "path",
      {
        "d": "M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "redo-2" },
    $$sanitized_props,
    {
      /**
       * @component @name Redo2
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJtMTUgMTQgNS01LTUtNSIgLz4KICA8cGF0aCBkPSJNMjAgOUg5LjVBNS41IDUuNSAwIDAgMCA0IDE0LjVBNS41IDUuNSAwIDAgMCA5LjUgMjBIMTMiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/redo-2
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Refresh_cw($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      { "d": "M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" }
    ],
    ["path", { "d": "M21 3v5h-5" }],
    [
      "path",
      { "d": "M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" }
    ],
    ["path", { "d": "M8 16H3v5" }]
  ];
  Icon($$renderer, spread_props([
    { name: "refresh-cw" },
    $$sanitized_props,
    {
      /**
       * @component @name RefreshCw
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMyAxMmE5IDkgMCAwIDEgOS05IDkuNzUgOS43NSAwIDAgMSA2Ljc0IDIuNzRMMjEgOCIgLz4KICA8cGF0aCBkPSJNMjEgM3Y1aC01IiAvPgogIDxwYXRoIGQ9Ik0yMSAxMmE5IDkgMCAwIDEtOSA5IDkuNzUgOS43NSAwIDAgMS02Ljc0LTIuNzRMMyAxNiIgLz4KICA8cGF0aCBkPSJNOCAxNkgzdjUiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/refresh-cw
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Rotate_ccw($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      { "d": "M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" }
    ],
    ["path", { "d": "M3 3v5h5" }]
  ];
  Icon($$renderer, spread_props([
    { name: "rotate-ccw" },
    $$sanitized_props,
    {
      /**
       * @component @name RotateCcw
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMyAxMmE5IDkgMCAxIDAgOS05IDkuNzUgOS43NSAwIDAgMC02Ljc0IDIuNzRMMyA4IiAvPgogIDxwYXRoIGQ9Ik0zIDN2NWg1IiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/rotate-ccw
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Trash_2($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M3 6h18" }],
    ["path", { "d": "M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" }],
    ["path", { "d": "M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" }],
    ["line", { "x1": "10", "x2": "10", "y1": "11", "y2": "17" }],
    ["line", { "x1": "14", "x2": "14", "y1": "11", "y2": "17" }]
  ];
  Icon($$renderer, spread_props([
    { name: "trash-2" },
    $$sanitized_props,
    {
      /**
       * @component @name Trash2
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMyA2aDE4IiAvPgogIDxwYXRoIGQ9Ik0xOSA2djE0YzAgMS0xIDItMiAySDdjLTEgMC0yLTEtMi0yVjYiIC8+CiAgPHBhdGggZD0iTTggNlY0YzAtMSAxLTIgMi0yaDRjMSAwIDIgMSAyIDJ2MiIgLz4KICA8bGluZSB4MT0iMTAiIHgyPSIxMCIgeTE9IjExIiB5Mj0iMTciIC8+CiAgPGxpbmUgeDE9IjE0IiB4Mj0iMTQiIHkxPSIxMSIgeTI9IjE3IiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/trash-2
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Undo_2($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M9 14 4 9l5-5" }],
    [
      "path",
      {
        "d": "M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "undo-2" },
    $$sanitized_props,
    {
      /**
       * @component @name Undo2
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNOSAxNCA0IDlsNS01IiAvPgogIDxwYXRoIGQ9Ik00IDloMTAuNWE1LjUgNS41IDAgMCAxIDUuNSA1LjVhNS41IDUuNSAwIDAgMS01LjUgNS41SDExIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/undo-2
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Wand_sparkles($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72"
      }
    ],
    ["path", { "d": "m14 7 3 3" }],
    ["path", { "d": "M5 6v4" }],
    ["path", { "d": "M19 14v4" }],
    ["path", { "d": "M10 2v2" }],
    ["path", { "d": "M7 8H3" }],
    ["path", { "d": "M21 16h-4" }],
    ["path", { "d": "M11 3H9" }]
  ];
  Icon($$renderer, spread_props([
    { name: "wand-sparkles" },
    $$sanitized_props,
    {
      /**
       * @component @name WandSparkles
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJtMjEuNjQgMy42NC0xLjI4LTEuMjhhMS4yMSAxLjIxIDAgMCAwLTEuNzIgMEwyLjM2IDE4LjY0YTEuMjEgMS4yMSAwIDAgMCAwIDEuNzJsMS4yOCAxLjI4YTEuMiAxLjIgMCAwIDAgMS43MiAwTDIxLjY0IDUuMzZhMS4yIDEuMiAwIDAgMCAwLTEuNzIiIC8+CiAgPHBhdGggZD0ibTE0IDcgMyAzIiAvPgogIDxwYXRoIGQ9Ik01IDZ2NCIgLz4KICA8cGF0aCBkPSJNMTkgMTR2NCIgLz4KICA8cGF0aCBkPSJNMTAgMnYyIiAvPgogIDxwYXRoIGQ9Ik03IDhIMyIgLz4KICA8cGF0aCBkPSJNMjEgMTZoLTQiIC8+CiAgPHBhdGggZD0iTTExIDNIOSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/wand-sparkles
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
const dialogAttrs = createBitsAttrs({
  component: "dialog",
  parts: [
    "content",
    "trigger",
    "overlay",
    "title",
    "description",
    "close",
    "cancel",
    "action"
  ]
});
const DialogRootContext = new Context("Dialog.Root | AlertDialog.Root");
class DialogRootState {
  static create(opts) {
    const parent = DialogRootContext.getOr(null);
    return DialogRootContext.set(new DialogRootState(opts, parent));
  }
  opts;
  triggerNode = null;
  contentNode = null;
  overlayNode = null;
  descriptionNode = null;
  contentId = void 0;
  titleId = void 0;
  triggerId = void 0;
  descriptionId = void 0;
  cancelNode = null;
  nestedOpenCount = 0;
  depth;
  parent;
  contentPresence;
  overlayPresence;
  constructor(opts, parent) {
    this.opts = opts;
    this.parent = parent;
    this.depth = parent ? parent.depth + 1 : 0;
    this.handleOpen = this.handleOpen.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.contentPresence = new PresenceManager({
      ref: boxWith(() => this.contentNode),
      open: this.opts.open,
      enabled: true,
      onComplete: () => {
        this.opts.onOpenChangeComplete.current(this.opts.open.current);
      }
    });
    this.overlayPresence = new PresenceManager({
      ref: boxWith(() => this.overlayNode),
      open: this.opts.open,
      enabled: true
    });
    watch(
      () => this.opts.open.current,
      (isOpen) => {
        if (!this.parent) return;
        if (isOpen) {
          this.parent.incrementNested();
        } else {
          this.parent.decrementNested();
        }
      },
      { lazy: true }
    );
  }
  handleOpen() {
    if (this.opts.open.current) return;
    this.opts.open.current = true;
  }
  handleClose() {
    if (!this.opts.open.current) return;
    this.opts.open.current = false;
  }
  getBitsAttr = (part) => {
    return dialogAttrs.getAttr(part, this.opts.variant.current);
  };
  incrementNested() {
    this.nestedOpenCount++;
    this.parent?.incrementNested();
  }
  decrementNested() {
    if (this.nestedOpenCount === 0) return;
    this.nestedOpenCount--;
    this.parent?.decrementNested();
  }
  #sharedProps = derived(() => ({ "data-state": getDataOpenClosed(this.opts.open.current) }));
  get sharedProps() {
    return this.#sharedProps();
  }
  set sharedProps($$value) {
    return this.#sharedProps($$value);
  }
}
class DialogTitleState {
  static create(opts) {
    return new DialogTitleState(opts, DialogRootContext.get());
  }
  opts;
  root;
  attachment;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.root.titleId = this.opts.id.current;
    this.attachment = attachRef(this.opts.ref);
    watch.pre(() => this.opts.id.current, (id) => {
      this.root.titleId = id;
    });
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: "heading",
    "aria-level": this.opts.level.current,
    [this.root.getBitsAttr("title")]: "",
    ...this.root.sharedProps,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class DialogContentState {
  static create(opts) {
    return new DialogContentState(opts, DialogRootContext.get());
  }
  opts;
  root;
  attachment;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref, (v2) => {
      this.root.contentNode = v2;
      this.root.contentId = v2?.id;
    });
  }
  #snippetProps = derived(() => ({ open: this.root.opts.open.current }));
  get snippetProps() {
    return this.#snippetProps();
  }
  set snippetProps($$value) {
    return this.#snippetProps($$value);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: this.root.opts.variant.current === "alert-dialog" ? "alertdialog" : "dialog",
    "aria-modal": "true",
    "aria-describedby": this.root.descriptionId,
    "aria-labelledby": this.root.titleId,
    [this.root.getBitsAttr("content")]: "",
    style: {
      pointerEvents: "auto",
      outline: this.root.opts.variant.current === "alert-dialog" ? "none" : void 0,
      "--bits-dialog-depth": this.root.depth,
      "--bits-dialog-nested-count": this.root.nestedOpenCount,
      contain: "layout style"
    },
    tabindex: this.root.opts.variant.current === "alert-dialog" ? -1 : void 0,
    "data-nested-open": boolToEmptyStrOrUndef(this.root.nestedOpenCount > 0),
    "data-nested": boolToEmptyStrOrUndef(this.root.parent !== null),
    ...getDataTransitionAttrs(this.root.contentPresence.transitionStatus),
    ...this.root.sharedProps,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
  get shouldRender() {
    return this.root.contentPresence.shouldRender;
  }
}
class DialogOverlayState {
  static create(opts) {
    return new DialogOverlayState(opts, DialogRootContext.get());
  }
  opts;
  root;
  attachment;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref, (v2) => this.root.overlayNode = v2);
  }
  #snippetProps = derived(() => ({ open: this.root.opts.open.current }));
  get snippetProps() {
    return this.#snippetProps();
  }
  set snippetProps($$value) {
    return this.#snippetProps($$value);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    [this.root.getBitsAttr("overlay")]: "",
    style: {
      pointerEvents: "auto",
      "--bits-dialog-depth": this.root.depth,
      "--bits-dialog-nested-count": this.root.nestedOpenCount
    },
    "data-nested-open": boolToEmptyStrOrUndef(this.root.nestedOpenCount > 0),
    "data-nested": boolToEmptyStrOrUndef(this.root.parent !== null),
    ...getDataTransitionAttrs(this.root.overlayPresence.transitionStatus),
    ...this.root.sharedProps,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
  get shouldRender() {
    return this.root.overlayPresence.shouldRender;
  }
}
function Dialog_title($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      child,
      children,
      level = 2,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const titleState = DialogTitleState.create({
      id: boxWith(() => id),
      level: boxWith(() => level),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, titleState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Dialog_overlay($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      forceMount = false,
      child,
      children,
      ref = null,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const overlayState = DialogOverlayState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, overlayState.props));
    if (overlayState.shouldRender || forceMount) {
      $$renderer2.push("<!--[0-->");
      if (child) {
        $$renderer2.push("<!--[0-->");
        child($$renderer2, {
          props: mergeProps(mergedProps()),
          ...overlayState.snippetProps
        });
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<div${attributes({ ...mergeProps(mergedProps()) })}>`);
        children?.($$renderer2, overlayState.snippetProps);
        $$renderer2.push(`<!----></div>`);
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function findNextSibling(el, selector) {
  let sibling = el.nextElementSibling;
  while (sibling) {
    if (sibling.matches(selector))
      return sibling;
    sibling = sibling.nextElementSibling;
  }
}
function findPreviousSibling(el, selector) {
  let sibling = el.previousElementSibling;
  while (sibling) {
    if (sibling.matches(selector))
      return sibling;
    sibling = sibling.previousElementSibling;
  }
}
function cssEscape(value) {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  const length = value.length;
  let index = -1;
  let codeUnit;
  let result = "";
  const firstCodeUnit = value.charCodeAt(0);
  if (length === 1 && firstCodeUnit === 45)
    return "\\" + value;
  while (++index < length) {
    codeUnit = value.charCodeAt(index);
    if (codeUnit === 0) {
      result += "�";
      continue;
    }
    if (
      // If the character is in the range [\1-\1F] (U+0001 to U+001F) or is U+007F
      codeUnit >= 1 && codeUnit <= 31 || codeUnit === 127 || // If the character is the first character and is in the range [0-9] (U+0030 to U+0039)
      index === 0 && codeUnit >= 48 && codeUnit <= 57 || // If the character is the second character and is in the range [0-9] (U+0030 to U+0039)
      // and the first character is a `-` (U+002D)
      index === 1 && codeUnit >= 48 && codeUnit <= 57 && firstCodeUnit === 45
    ) {
      result += "\\" + codeUnit.toString(16) + " ";
      continue;
    }
    if (codeUnit >= 128 || codeUnit === 45 || codeUnit === 95 || codeUnit >= 48 && codeUnit <= 57 || codeUnit >= 65 && codeUnit <= 90 || codeUnit >= 97 && codeUnit <= 122) {
      result += value.charAt(index);
      continue;
    }
    result += "\\" + value.charAt(index);
  }
  return result;
}
const COMMAND_VALUE_ATTR = "data-value";
const commandAttrs = createBitsAttrs({
  component: "command",
  parts: [
    "root",
    "list",
    "input",
    "separator",
    "loading",
    "empty",
    "group",
    "group-items",
    "group-heading",
    "item",
    "viewport",
    "input-label"
  ]
});
const COMMAND_GROUP_SELECTOR = commandAttrs.selector("group");
const COMMAND_GROUP_ITEMS_SELECTOR = commandAttrs.selector("group-items");
const COMMAND_GROUP_HEADING_SELECTOR = commandAttrs.selector("group-heading");
const COMMAND_ITEM_SELECTOR = commandAttrs.selector("item");
const COMMAND_VALID_ITEM_SELECTOR = `${commandAttrs.selector("item")}:not([aria-disabled="true"])`;
const CommandRootContext = new Context("Command.Root");
const CommandListContext = new Context("Command.List");
const CommandGroupContainerContext = new Context("Command.Group");
const defaultState = {
  search: "",
  value: "",
  filtered: { count: 0, items: /* @__PURE__ */ new Map(), groups: /* @__PURE__ */ new Set() }
};
class CommandRootState {
  static create(opts) {
    return CommandRootContext.set(new CommandRootState(opts));
  }
  opts;
  attachment;
  #updateScheduled = false;
  #isInitialMount = true;
  sortAfterTick = false;
  sortAndFilterAfterTick = false;
  allItems = /* @__PURE__ */ new Set();
  allGroups = /* @__PURE__ */ new Map();
  allIds = /* @__PURE__ */ new Map();
  // attempt to prevent the harsh delay when user is typing fast
  key = 0;
  viewportNode = null;
  inputNode = null;
  labelNode = null;
  // published state that the components and other things can react to
  commandState = defaultState;
  // internal state that we mutate in batches and publish to the `state` at once
  _commandState = defaultState;
  #snapshot() {
    return snapshot(this._commandState);
  }
  #scheduleUpdate() {
    if (this.#updateScheduled) return;
    this.#updateScheduled = true;
    afterTick(() => {
      this.#updateScheduled = false;
      const currentState = this.#snapshot();
      const hasStateChanged = !Object.is(this.commandState, currentState);
      if (hasStateChanged) {
        this.commandState = currentState;
        this.opts.onStateChange?.current?.(currentState);
      }
    });
  }
  setState(key2, value, preventScroll) {
    if (Object.is(this._commandState[key2], value)) return;
    this._commandState[key2] = value;
    if (key2 === "search") {
      this.#filterItems();
      this.#sort();
    } else if (key2 === "value") {
      if (!preventScroll) this.#scrollSelectedIntoView();
    }
    this.#scheduleUpdate();
  }
  constructor(opts) {
    this.opts = opts;
    this.attachment = attachRef(this.opts.ref);
    const defaults = { ...this._commandState, value: this.opts.value.current ?? "" };
    this._commandState = defaults;
    this.commandState = defaults;
    this.onkeydown = this.onkeydown.bind(this);
  }
  /**
   * Calculates score for an item based on search text and keywords.
   * Higher score = better match.
   *
   * @param value - Item's display text
   * @param keywords - Optional keywords to boost scoring
   * @returns Score from 0-1, where 0 = no match
   */
  #score(value, keywords) {
    const filter = this.opts.filter.current ?? computeCommandScore;
    const score = value ? filter(value, this._commandState.search, keywords) : 0;
    return score;
  }
  /**
   * Sorts items and groups based on search scores.
   * Groups are sorted by their highest scoring item.
   * When no search active, selects first item.
   */
  #sort() {
    if (!this._commandState.search || this.opts.shouldFilter.current === false) {
      if (!this._commandState.value || !this.#isInitialMount) {
        this.#selectFirstItem();
      } else if (this.#isInitialMount && this._commandState.value) {
        this.#scrollInitialValue();
      }
      return;
    }
    const scores = this._commandState.filtered.items;
    const groups = [];
    for (const value of this._commandState.filtered.groups) {
      const items = this.allGroups.get(value);
      let max = 0;
      if (!items) {
        groups.push([value, max]);
        continue;
      }
      for (const item of items) {
        const score = scores.get(item);
        max = Math.max(score ?? 0, max);
      }
      groups.push([value, max]);
    }
    const listInsertionElement = this.viewportNode;
    const sorted = this.getValidItems().sort((a, b) => {
      const valueA = a.getAttribute("data-value");
      const valueB = b.getAttribute("data-value");
      const scoresA = scores.get(valueA) ?? 0;
      const scoresB = scores.get(valueB) ?? 0;
      return scoresB - scoresA;
    });
    for (const item of sorted) {
      const group = item.closest(COMMAND_GROUP_ITEMS_SELECTOR);
      if (group) {
        const itemToAppend = item.parentElement === group ? item : item.closest(`${COMMAND_GROUP_ITEMS_SELECTOR} > *`);
        if (itemToAppend) {
          group.appendChild(itemToAppend);
        }
      } else {
        const itemToAppend = item.parentElement === listInsertionElement ? item : item.closest(`${COMMAND_GROUP_ITEMS_SELECTOR} > *`);
        if (itemToAppend) {
          listInsertionElement?.appendChild(itemToAppend);
        }
      }
    }
    const sortedGroups = groups.sort((a, b) => b[1] - a[1]);
    for (const group of sortedGroups) {
      const element = listInsertionElement?.querySelector(`${COMMAND_GROUP_SELECTOR}[${COMMAND_VALUE_ATTR}="${cssEscape(group[0])}"]`);
      element?.parentElement?.appendChild(element);
    }
    this.#selectFirstItem();
  }
  /**
   * Sets current value and triggers re-render if cleared.
   *
   * @param value - New value to set
   */
  setValue(value, opts) {
    if (value !== this.opts.value.current && value === "") {
      afterTick(() => {
        this.key++;
      });
    }
    this.setState("value", value, opts);
    this.opts.value.current = value;
  }
  /**
   * Selects first non-disabled item on next tick.
   */
  #selectFirstItem() {
    afterTick(() => {
      const item = this.getValidItems().find((item2) => item2.getAttribute("aria-disabled") !== "true");
      const value = item?.getAttribute(COMMAND_VALUE_ATTR);
      const shouldPreventScroll = this.#isInitialMount && this.opts.disableInitialScroll.current;
      this.setValue(value ?? "", shouldPreventScroll);
      this.#isInitialMount = false;
    });
  }
  /**
   * Scrolls the initial value into view if it exists and is not the first item.
   * Called during initial mount when a value is provided.
   */
  #scrollInitialValue() {
    afterTick(() => {
      const shouldPreventScroll = this.opts.disableInitialScroll.current;
      if (!shouldPreventScroll) {
        this.#scrollSelectedIntoView();
      }
      this.#isInitialMount = false;
    });
  }
  /**
   * Updates filtered items/groups based on search.
   * Recalculates scores and filtered count.
   */
  #filterItems() {
    if (!this._commandState.search || this.opts.shouldFilter.current === false) {
      this._commandState.filtered.count = this.allItems.size;
      return;
    }
    this._commandState.filtered.groups = /* @__PURE__ */ new Set();
    let itemCount = 0;
    for (const id of this.allItems) {
      const value = this.allIds.get(id)?.value ?? "";
      const keywords = this.allIds.get(id)?.keywords ?? [];
      const rank = this.#score(value, keywords);
      this._commandState.filtered.items.set(id, rank);
      if (rank > 0) itemCount++;
    }
    for (const [groupId, group] of this.allGroups) {
      for (const itemId of group) {
        const currItem = this._commandState.filtered.items.get(itemId);
        if (currItem && currItem > 0) {
          this._commandState.filtered.groups.add(groupId);
          break;
        }
      }
    }
    this._commandState.filtered.count = itemCount;
  }
  /**
   * Gets all non-disabled, visible command items.
   *
   * @returns Array of valid item elements
   * @remarks Exposed for direct item access and bound checking
   */
  getValidItems() {
    const node = this.opts.ref.current;
    if (!node) return [];
    const validItems = Array.from(node.querySelectorAll(COMMAND_VALID_ITEM_SELECTOR)).filter((el) => !!el);
    return validItems;
  }
  /**
   * Gets all visible command items.
   *
   * @returns Array of valid item elements
   * @remarks Exposed for direct item access and bound checking
   */
  getVisibleItems() {
    const node = this.opts.ref.current;
    if (!node) return [];
    const visibleItems = Array.from(node.querySelectorAll(COMMAND_ITEM_SELECTOR)).filter((el) => !!el);
    return visibleItems;
  }
  /** Returns all visible items in a matrix structure
   *
   * @remarks Returns empty if the command isn't configured as a grid
   *
   * @returns
   */
  get itemsGrid() {
    if (!this.isGrid) return [];
    const columns = this.opts.columns.current ?? 1;
    const items = this.getVisibleItems();
    const grid = [[]];
    let currentGroup = items[0]?.getAttribute("data-group");
    let column = 0;
    let row = 0;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const itemGroup = item?.getAttribute("data-group");
      if (currentGroup !== itemGroup) {
        currentGroup = itemGroup;
        column = 1;
        row++;
        grid.push([{ index: i, firstRowOfGroup: true, ref: item }]);
      } else {
        column++;
        if (column > columns) {
          row++;
          column = 1;
          grid.push([]);
        }
        grid[row]?.push({
          index: i,
          firstRowOfGroup: grid[row]?.[0]?.firstRowOfGroup ?? i === 0,
          ref: item
        });
      }
    }
    return grid;
  }
  /**
   * Gets currently selected command item.
   *
   * @returns Selected element or undefined
   */
  #getSelectedItem() {
    const node = this.opts.ref.current;
    if (!node) return;
    const selectedNode = node.querySelector(`${COMMAND_VALID_ITEM_SELECTOR}[data-selected]`);
    if (!selectedNode) return;
    return selectedNode;
  }
  /**
   * Scrolls selected item into view.
   * Special handling for first items in groups.
   */
  #scrollSelectedIntoView() {
    afterTick(() => {
      const item = this.#getSelectedItem();
      if (!item) return;
      const grandparent = item.parentElement?.parentElement;
      if (!grandparent) return;
      if (this.isGrid) {
        const isFirstRowOfGroup = this.#itemIsFirstRowOfGroup(item);
        item.scrollIntoView({ block: "nearest" });
        if (isFirstRowOfGroup) {
          const closestGroupHeader = item?.closest(COMMAND_GROUP_SELECTOR)?.querySelector(COMMAND_GROUP_HEADING_SELECTOR);
          closestGroupHeader?.scrollIntoView({ block: "nearest" });
          return;
        }
      } else {
        const firstChildOfParent = getFirstNonCommentChild(grandparent);
        if (firstChildOfParent && firstChildOfParent.dataset?.value === item.dataset?.value) {
          const closestGroupHeader = item?.closest(COMMAND_GROUP_SELECTOR)?.querySelector(COMMAND_GROUP_HEADING_SELECTOR);
          closestGroupHeader?.scrollIntoView({ block: "nearest" });
          return;
        }
      }
      item.scrollIntoView({ block: "nearest" });
    });
  }
  #itemIsFirstRowOfGroup(item) {
    const grid = this.itemsGrid;
    if (grid.length === 0) return false;
    for (let r = 0; r < grid.length; r++) {
      const row = grid[r];
      if (row === void 0) continue;
      for (let c = 0; c < row.length; c++) {
        const column = row[c];
        if (column === void 0 || column.ref !== item) continue;
        return column.firstRowOfGroup;
      }
    }
    return false;
  }
  /**
   * Sets selection to item at specified index in valid items array.
   * If index is out of bounds, does nothing.
   *
   * @param index - Zero-based index of item to select
   * @remarks
   * Uses `getValidItems()` to get selectable items, filtering out disabled/hidden ones.
   * Access valid items directly via `getValidItems()` to check bounds before calling.
   *
   * @example
   * // get valid items length for bounds check
   * const items = getValidItems()
   * if (index < items.length) {
   *   updateSelectedToIndex(index)
   * }
   */
  updateSelectedToIndex(index) {
    const item = this.getValidItems()[index];
    if (!item) return;
    this.setValue(item.getAttribute(COMMAND_VALUE_ATTR) ?? "");
  }
  /**
   * Updates selected item by moving up/down relative to current selection.
   * Handles wrapping when loop option is enabled.
   *
   * @param change - Direction to move: 1 for next item, -1 for previous item
   * @remarks
   * The loop behavior wraps:
   * - From last item to first when moving next
   * - From first item to last when moving previous
   *
   * Uses `getValidItems()` to get all selectable items, which filters out disabled/hidden items.
   * You can call `getValidItems()` directly to get the current valid items array.
   *
   * @example
   * // select next item
   * updateSelectedByItem(1)
   *
   * // get all valid items
   * const items = getValidItems()
   */
  updateSelectedByItem(change) {
    const selected = this.#getSelectedItem();
    const items = this.getValidItems();
    const index = items.findIndex((item) => item === selected);
    let newSelected = items[index + change];
    if (this.opts.loop.current) {
      newSelected = index + change < 0 ? items[items.length - 1] : index + change === items.length ? items[0] : items[index + change];
    }
    if (newSelected) {
      this.setValue(newSelected.getAttribute(COMMAND_VALUE_ATTR) ?? "");
    }
  }
  /**
   * Moves selection to the first valid item in the next/previous group.
   * If no group is found, falls back to selecting the next/previous item globally.
   *
   * @param change - Direction to move: 1 for next group, -1 for previous group
   * @example
   * // move to first item in next group
   * updateSelectedByGroup(1)
   *
   * // move to first item in previous group
   * updateSelectedByGroup(-1)
   */
  updateSelectedByGroup(change) {
    const selected = this.#getSelectedItem();
    let group = selected?.closest(COMMAND_GROUP_SELECTOR);
    let item;
    while (group && !item) {
      group = change > 0 ? findNextSibling(group, COMMAND_GROUP_SELECTOR) : findPreviousSibling(group, COMMAND_GROUP_SELECTOR);
      item = group?.querySelector(COMMAND_VALID_ITEM_SELECTOR);
    }
    if (item) {
      this.setValue(item.getAttribute(COMMAND_VALUE_ATTR) ?? "");
    } else {
      this.updateSelectedByItem(change);
    }
  }
  /**
   * Maps item id to display value and search keywords.
   * Returns cleanup function to remove mapping.
   *
   * @param id - Unique item identifier
   * @param value - Display text
   * @param keywords - Optional search boost terms
   * @returns Cleanup function
   */
  registerValue(value, keywords) {
    if (!(value && value === this.allIds.get(value)?.value)) {
      this.allIds.set(value, { value, keywords });
    }
    this._commandState.filtered.items.set(value, this.#score(value, keywords));
    if (!this.sortAfterTick) {
      this.sortAfterTick = true;
      afterTick(() => {
        this.#sort();
        this.sortAfterTick = false;
      });
    }
    return () => {
      this.allIds.delete(value);
    };
  }
  /**
   * Registers item in command list and its group.
   * Handles filtering, sorting and selection updates.
   *
   * @param id - Item identifier
   * @param groupId - Optional group to add item to
   * @returns Cleanup function that handles selection
   */
  registerItem(id, groupId) {
    this.allItems.add(id);
    if (groupId) {
      if (!this.allGroups.has(groupId)) {
        this.allGroups.set(groupId, /* @__PURE__ */ new Set([id]));
      } else {
        this.allGroups.get(groupId).add(id);
      }
    }
    if (!this.sortAndFilterAfterTick) {
      this.sortAndFilterAfterTick = true;
      afterTick(() => {
        this.#filterItems();
        this.#sort();
        this.sortAndFilterAfterTick = false;
      });
    }
    this.#scheduleUpdate();
    return () => {
      const selectedItem = this.#getSelectedItem();
      this.allItems.delete(id);
      this.commandState.filtered.items.delete(id);
      this.#filterItems();
      if (selectedItem?.getAttribute("id") === id) {
        this.#selectFirstItem();
      }
      this.#scheduleUpdate();
    };
  }
  /**
   * Creates empty group if not exists.
   *
   * @param id - Group identifier
   * @returns Cleanup function
   */
  registerGroup(id) {
    if (!this.allGroups.has(id)) {
      this.allGroups.set(id, /* @__PURE__ */ new Set());
    }
    return () => {
      this.allIds.delete(id);
      this.allGroups.delete(id);
    };
  }
  get isGrid() {
    return this.opts.columns.current !== null;
  }
  /**
   * Selects last valid item.
   */
  #last() {
    return this.updateSelectedToIndex(this.getValidItems().length - 1);
  }
  /**
   * Handles next item selection:
   * - Meta: Jump to last
   * - Alt: Next group
   * - Default: Next item
   *
   * @param e - Keyboard event
   */
  #next(e) {
    e.preventDefault();
    if (e.metaKey) {
      this.#last();
    } else if (e.altKey) {
      this.updateSelectedByGroup(1);
    } else {
      this.updateSelectedByItem(1);
    }
  }
  #down(e) {
    if (this.opts.columns.current === null) return;
    e.preventDefault();
    if (e.metaKey) {
      this.updateSelectedByGroup(1);
    } else {
      this.updateSelectedByItem(this.#nextRowColumnOffset(e));
    }
  }
  #getColumn(item, grid) {
    if (grid.length === 0) return null;
    for (let r = 0; r < grid.length; r++) {
      const row = grid[r];
      if (row === void 0) continue;
      for (let c = 0; c < row.length; c++) {
        const column = row[c];
        if (column === void 0 || column.ref !== item) continue;
        return { columnIndex: c, rowIndex: r };
      }
    }
    return null;
  }
  #nextRowColumnOffset(e) {
    const grid = this.itemsGrid;
    const selected = this.#getSelectedItem();
    if (!selected) return 0;
    const column = this.#getColumn(selected, grid);
    if (!column) return 0;
    let newItem = null;
    const skipRows = e.altKey ? 1 : 0;
    if (e.altKey && column.rowIndex === grid.length - 2 && !this.opts.loop.current) {
      newItem = this.#findNextNonDisabledItem({
        start: grid.length - 1,
        end: grid.length,
        expectedColumnIndex: column.columnIndex,
        grid
      });
    } else if (column.rowIndex === grid.length - 1) {
      if (!this.opts.loop.current) return 0;
      newItem = this.#findNextNonDisabledItem({
        start: 0 + skipRows,
        end: column.rowIndex,
        expectedColumnIndex: column.columnIndex,
        grid
      });
    } else {
      newItem = this.#findNextNonDisabledItem({
        start: column.rowIndex + 1 + skipRows,
        end: grid.length,
        expectedColumnIndex: column.columnIndex,
        grid
      });
      if (newItem === null && this.opts.loop.current) {
        newItem = this.#findNextNonDisabledItem({
          start: 0,
          end: column.rowIndex,
          expectedColumnIndex: column.columnIndex,
          grid
        });
      }
    }
    return this.#calculateOffset(selected, newItem);
  }
  /** Attempts to find the next non-disabled column that matches the expected column.
   *
   * @remarks
   * - Skips over disabled columns
   * - When a row is shorter than the expected column it defaults to the last item in the row
   *
   * @param param0
   * @returns
   */
  #findNextNonDisabledItem({ start, end, grid, expectedColumnIndex }) {
    let newItem = null;
    for (let r = start; r < end; r++) {
      const row = grid[r];
      newItem = row[expectedColumnIndex]?.ref ?? null;
      if (newItem !== null && itemIsDisabled(newItem)) {
        newItem = null;
        continue;
      }
      if (newItem === null) {
        for (let i = row.length - 1; i >= 0; i--) {
          const item = row[row.length - 1];
          if (item === void 0 || itemIsDisabled(item.ref)) continue;
          newItem = item.ref;
          break;
        }
      }
      break;
    }
    return newItem;
  }
  #calculateOffset(selected, newSelected) {
    if (newSelected === null) return 0;
    const items = this.getValidItems();
    const ogIndex = items.findIndex((item) => item === selected);
    const newIndex = items.findIndex((item) => item === newSelected);
    return newIndex - ogIndex;
  }
  #up(e) {
    if (this.opts.columns.current === null) return;
    e.preventDefault();
    if (e.metaKey) {
      this.updateSelectedByGroup(-1);
    } else {
      this.updateSelectedByItem(this.#previousRowColumnOffset(e));
    }
  }
  #previousRowColumnOffset(e) {
    const grid = this.itemsGrid;
    const selected = this.#getSelectedItem();
    if (selected === void 0) return 0;
    const column = this.#getColumn(selected, grid);
    if (column === null) return 0;
    let newItem = null;
    const skipRows = e.altKey ? 1 : 0;
    if (e.altKey && column.rowIndex === 1 && this.opts.loop.current === false) {
      newItem = this.#findNextNonDisabledItemDesc({
        start: 0,
        end: 0,
        expectedColumnIndex: column.columnIndex,
        grid
      });
    } else if (column.rowIndex === 0) {
      if (this.opts.loop.current === false) return 0;
      newItem = this.#findNextNonDisabledItemDesc({
        start: grid.length - 1 - skipRows,
        end: column.rowIndex + 1,
        expectedColumnIndex: column.columnIndex,
        grid
      });
    } else {
      newItem = this.#findNextNonDisabledItemDesc({
        start: column.rowIndex - 1 - skipRows,
        end: 0,
        expectedColumnIndex: column.columnIndex,
        grid
      });
      if (newItem === null && this.opts.loop.current) {
        newItem = this.#findNextNonDisabledItemDesc({
          start: grid.length - 1,
          end: column.rowIndex + 1,
          expectedColumnIndex: column.columnIndex,
          grid
        });
      }
    }
    return this.#calculateOffset(selected, newItem);
  }
  /**
   * Attempts to find the next non-disabled column that matches the expected column.
   *
   * @remarks
   * - Skips over disabled columns
   * - When a row is shorter than the expected column it defaults to the last item in the row
   */
  #findNextNonDisabledItemDesc({ start, end, grid, expectedColumnIndex }) {
    let newItem = null;
    for (let r = start; r >= end; r--) {
      const row = grid[r];
      if (row === void 0) continue;
      newItem = row[expectedColumnIndex]?.ref ?? null;
      if (newItem !== null && itemIsDisabled(newItem)) {
        newItem = null;
        continue;
      }
      if (newItem === null) {
        for (let i = row.length - 1; i >= 0; i--) {
          const item = row[row.length - 1];
          if (item === void 0 || itemIsDisabled(item.ref)) continue;
          newItem = item.ref;
          break;
        }
      }
      break;
    }
    return newItem;
  }
  /**
   * Handles previous item selection:
   * - Meta: Jump to first
   * - Alt: Previous group
   * - Default: Previous item
   *
   * @param e - Keyboard event
   */
  #prev(e) {
    e.preventDefault();
    if (e.metaKey) {
      this.updateSelectedToIndex(0);
    } else if (e.altKey) {
      this.updateSelectedByGroup(-1);
    } else {
      this.updateSelectedByItem(-1);
    }
  }
  onkeydown(e) {
    const isVim = this.opts.vimBindings.current && e.ctrlKey;
    switch (e.key) {
      case n:
      case j: {
        if (isVim) {
          if (this.isGrid) {
            this.#down(e);
          } else {
            this.#next(e);
          }
        }
        break;
      }
      case l: {
        if (isVim) {
          if (this.isGrid) {
            this.#next(e);
          }
        }
        break;
      }
      case ARROW_DOWN:
        if (this.isGrid) {
          this.#down(e);
        } else {
          this.#next(e);
        }
        break;
      case ARROW_RIGHT:
        if (!this.isGrid) break;
        this.#next(e);
        break;
      case p:
      case k: {
        if (isVim) {
          if (this.isGrid) {
            this.#up(e);
          } else {
            this.#prev(e);
          }
        }
        break;
      }
      case h: {
        if (isVim && this.isGrid) {
          this.#prev(e);
        }
        break;
      }
      case ARROW_UP:
        if (this.isGrid) {
          this.#up(e);
        } else {
          this.#prev(e);
        }
        break;
      case ARROW_LEFT:
        if (!this.isGrid) break;
        this.#prev(e);
        break;
      case HOME:
        e.preventDefault();
        this.updateSelectedToIndex(0);
        break;
      case END:
        e.preventDefault();
        this.#last();
        break;
      case ENTER: {
        if (!e.isComposing && e.keyCode !== 229) {
          e.preventDefault();
          const item = this.#getSelectedItem();
          if (item) {
            item?.click();
          }
        }
      }
    }
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: "application",
    [commandAttrs.root]: "",
    tabindex: -1,
    onkeydown: this.onkeydown,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
function itemIsDisabled(item) {
  return item.getAttribute("aria-disabled") === "true";
}
class CommandEmptyState {
  static create(opts) {
    return new CommandEmptyState(opts, CommandRootContext.get());
  }
  opts;
  root;
  attachment;
  #shouldRender = derived(() => {
    return this.root._commandState.filtered.count === 0 && this.#isInitialRender === false || this.opts.forceMount.current;
  });
  get shouldRender() {
    return this.#shouldRender();
  }
  set shouldRender($$value) {
    return this.#shouldRender($$value);
  }
  #isInitialRender = true;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: "presentation",
    [commandAttrs.empty]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandGroupContainerState {
  static create(opts) {
    return CommandGroupContainerContext.set(new CommandGroupContainerState(opts, CommandRootContext.get()));
  }
  opts;
  root;
  attachment;
  #shouldRender = derived(() => {
    if (this.opts.forceMount.current) return true;
    if (this.root.opts.shouldFilter.current === false) return true;
    if (!this.root.commandState.search) return true;
    return this.root._commandState.filtered.groups.has(this.trueValue);
  });
  get shouldRender() {
    return this.#shouldRender();
  }
  set shouldRender($$value) {
    return this.#shouldRender($$value);
  }
  headingNode = null;
  trueValue = "";
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref);
    this.trueValue = opts.value.current ?? opts.id.current;
    watch(() => this.trueValue, () => {
      return this.root.registerGroup(this.trueValue);
    });
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: "presentation",
    hidden: this.shouldRender ? void 0 : true,
    "data-value": this.trueValue,
    [commandAttrs.group]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandGroupHeadingState {
  static create(opts) {
    return new CommandGroupHeadingState(opts, CommandGroupContainerContext.get());
  }
  opts;
  group;
  attachment;
  constructor(opts, group) {
    this.opts = opts;
    this.group = group;
    this.attachment = attachRef(this.opts.ref, (v2) => this.group.headingNode = v2);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    [commandAttrs["group-heading"]]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandGroupItemsState {
  static create(opts) {
    return new CommandGroupItemsState(opts, CommandGroupContainerContext.get());
  }
  opts;
  group;
  attachment;
  constructor(opts, group) {
    this.opts = opts;
    this.group = group;
    this.attachment = attachRef(this.opts.ref);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: "group",
    [commandAttrs["group-items"]]: "",
    "aria-labelledby": this.group.headingNode?.id ?? void 0,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandInputState {
  static create(opts) {
    return new CommandInputState(opts, CommandRootContext.get());
  }
  opts;
  root;
  attachment;
  #selectedItemId = derived(() => {
    const item = this.root.viewportNode?.querySelector(`${COMMAND_ITEM_SELECTOR}[${COMMAND_VALUE_ATTR}="${cssEscape(this.root.opts.value.current)}"]`);
    if (item === void 0 || item === null) return;
    return item.getAttribute("id") ?? void 0;
  });
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref, (v2) => this.root.inputNode = v2);
    watch(() => this.opts.ref.current, () => {
      const node = this.opts.ref.current;
      if (node && this.opts.autofocus.current) {
        afterSleep(10, () => node.focus());
      }
    });
    watch(() => this.opts.value.current, () => {
      if (this.root.commandState.search !== this.opts.value.current) {
        this.root.setState("search", this.opts.value.current);
      }
    });
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    type: "text",
    [commandAttrs.input]: "",
    autocomplete: "off",
    autocorrect: "off",
    spellcheck: false,
    "aria-autocomplete": "list",
    role: "combobox",
    "aria-expanded": boolToStr(true),
    "aria-controls": this.root.viewportNode?.id ?? void 0,
    "aria-labelledby": this.root.labelNode?.id ?? void 0,
    "aria-activedescendant": this.#selectedItemId(),
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandItemState {
  static create(opts) {
    const group = CommandGroupContainerContext.getOr(null);
    return new CommandItemState({ ...opts, group }, CommandRootContext.get());
  }
  opts;
  root;
  attachment;
  #group = null;
  #trueForceMount = derived(() => {
    return this.opts.forceMount.current || this.#group?.opts.forceMount.current === true;
  });
  #shouldRender = derived(() => {
    this.opts.ref.current;
    if (this.#trueForceMount() || this.root.opts.shouldFilter.current === false || !this.root.commandState.search) {
      return true;
    }
    const currentScore = this.root.commandState.filtered.items.get(this.trueValue);
    if (currentScore === void 0) return false;
    return currentScore > 0;
  });
  get shouldRender() {
    return this.#shouldRender();
  }
  set shouldRender($$value) {
    return this.#shouldRender($$value);
  }
  #isSelected = derived(() => this.root.opts.value.current === this.trueValue && this.trueValue !== "");
  get isSelected() {
    return this.#isSelected();
  }
  set isSelected($$value) {
    return this.#isSelected($$value);
  }
  trueValue = "";
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.#group = CommandGroupContainerContext.getOr(null);
    this.trueValue = opts.value.current;
    this.attachment = attachRef(this.opts.ref);
    watch(
      [
        () => this.trueValue,
        () => this.#group?.trueValue,
        () => this.opts.forceMount.current
      ],
      () => {
        if (this.opts.forceMount.current || !this.trueValue) return;
        return this.root.registerItem(this.trueValue, this.#group?.trueValue);
      }
    );
    watch([() => this.opts.value.current, () => this.opts.ref.current], () => {
      if (this.opts.value.current) {
        this.trueValue = this.opts.value.current;
      } else if (this.opts.ref.current?.textContent) {
        this.trueValue = this.opts.ref.current.textContent.trim();
      }
      if (this.trueValue) {
        this.root.registerValue(this.trueValue, opts.keywords.current.map((kw) => kw.trim()));
        this.opts.ref.current?.setAttribute(COMMAND_VALUE_ATTR, this.trueValue);
      }
    });
    this.onclick = this.onclick.bind(this);
    this.onpointermove = this.onpointermove.bind(this);
  }
  #onSelect() {
    if (this.opts.disabled.current) return;
    this.#select();
    this.opts.onSelect?.current();
  }
  #select() {
    if (this.opts.disabled.current) return;
    this.root.setValue(this.trueValue, true);
  }
  onpointermove(_) {
    if (this.opts.disabled.current || this.root.opts.disablePointerSelection.current) return;
    this.#select();
  }
  onclick(_) {
    if (this.opts.disabled.current) return;
    this.#onSelect();
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    "aria-disabled": boolToStr(this.opts.disabled.current),
    "aria-selected": boolToStr(this.isSelected),
    "data-disabled": boolToEmptyStrOrUndef(this.opts.disabled.current),
    "data-selected": boolToEmptyStrOrUndef(this.isSelected),
    "data-value": this.trueValue,
    "data-group": this.#group?.trueValue,
    [commandAttrs.item]: "",
    role: "option",
    onpointermove: this.onpointermove,
    onclick: this.onclick,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandSeparatorState {
  static create(opts) {
    return new CommandSeparatorState(opts, CommandRootContext.get());
  }
  opts;
  root;
  attachment;
  #shouldRender = derived(() => !this.root._commandState.search || this.opts.forceMount.current);
  get shouldRender() {
    return this.#shouldRender();
  }
  set shouldRender($$value) {
    return this.#shouldRender($$value);
  }
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    // role="separator" cannot belong to a role="listbox"
    "aria-hidden": "true",
    [commandAttrs.separator]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandListState {
  static create(opts) {
    return CommandListContext.set(new CommandListState(opts, CommandRootContext.get()));
  }
  opts;
  root;
  attachment;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: "listbox",
    "aria-label": this.opts.ariaLabel.current,
    [commandAttrs.list]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandLabelState {
  static create(opts) {
    return new CommandLabelState(opts, CommandRootContext.get());
  }
  opts;
  root;
  attachment;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref, (v2) => this.root.labelNode = v2);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    [commandAttrs["input-label"]]: "",
    for: this.opts.for?.current,
    style: srOnlyStyles,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class CommandViewportState {
  static create(opts) {
    return new CommandViewportState(opts, CommandListContext.get());
  }
  opts;
  list;
  attachment;
  constructor(opts, list) {
    this.opts = opts;
    this.list = list;
    this.attachment = attachRef(this.opts.ref, (v2) => this.list.root.viewportNode = v2);
    watch(
      [
        () => this.opts.ref.current,
        () => this.list.opts.ref.current
      ],
      ([node, listNode]) => {
        if (node === null || listNode === null) return;
        let aF;
        const observer = new ResizeObserver(() => {
          aF = requestAnimationFrame(() => {
            const height = node.offsetHeight;
            listNode.style.setProperty("--bits-command-list-height", `${height.toFixed(1)}px`);
          });
        });
        observer.observe(node);
        return () => {
          cancelAnimationFrame(aF);
          observer.unobserve(node);
        };
      }
    );
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    [commandAttrs.viewport]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
function _command_label($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      children,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const labelState = CommandLabelState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, labelState.props));
    $$renderer2.push(`<label${attributes({ ...mergedProps() })}>`);
    children?.($$renderer2);
    $$renderer2.push(`<!----></label>`);
    bind_props($$props, { ref });
  });
}
function Command($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      value = "",
      onValueChange = noop,
      onStateChange = noop,
      loop = false,
      shouldFilter = true,
      filter = computeCommandScore,
      label = "",
      vimBindings = true,
      disablePointerSelection = false,
      disableInitialScroll = false,
      columns = null,
      children,
      child,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const rootState = CommandRootState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      filter: boxWith(() => filter),
      shouldFilter: boxWith(() => shouldFilter),
      loop: boxWith(() => loop),
      value: boxWith(() => value, (v2) => {
        if (value !== v2) {
          value = v2;
          onValueChange(v2);
        }
      }),
      vimBindings: boxWith(() => vimBindings),
      disablePointerSelection: boxWith(() => disablePointerSelection),
      disableInitialScroll: boxWith(() => disableInitialScroll),
      onStateChange: boxWith(() => onStateChange),
      columns: boxWith(() => columns)
    });
    const updateSelectedToIndex = (i) => rootState.updateSelectedToIndex(i);
    const updateSelectedByGroup = (c) => rootState.updateSelectedByGroup(c);
    const updateSelectedByItem = (c) => rootState.updateSelectedByItem(c);
    const getValidItems = () => rootState.getValidItems();
    const mergedProps = derived(() => mergeProps(restProps, rootState.props));
    function Label($$renderer3) {
      _command_label($$renderer3, {
        children: ($$renderer4) => {
          $$renderer4.push(`<!---->${escape_html(label)}`);
        },
        $$slots: { default: true }
      });
    }
    if (child) {
      $$renderer2.push("<!--[0-->");
      Label($$renderer2);
      $$renderer2.push(`<!----> `);
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
      Label($$renderer2);
      $$renderer2.push(`<!----> `);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      ref,
      value,
      updateSelectedToIndex,
      updateSelectedByGroup,
      updateSelectedByItem,
      getValidItems
    });
  });
}
function Command_empty($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      children,
      child,
      forceMount = false,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const emptyState = CommandEmptyState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      forceMount: boxWith(() => forceMount)
    });
    const mergedProps = derived(() => mergeProps(emptyState.props, restProps));
    if (emptyState.shouldRender) {
      $$renderer2.push("<!--[0-->");
      if (child) {
        $$renderer2.push("<!--[0-->");
        child($$renderer2, { props: mergedProps() });
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
        children?.($$renderer2);
        $$renderer2.push(`<!----></div>`);
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Command_group($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      value = "",
      forceMount = false,
      children,
      child,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const groupState = CommandGroupContainerState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      forceMount: boxWith(() => forceMount),
      value: boxWith(() => value)
    });
    const mergedProps = derived(() => mergeProps(restProps, groupState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Command_group_heading($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      children,
      child,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const headingState = CommandGroupHeadingState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, headingState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Command_group_items($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      children,
      child,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const groupItemsState = CommandGroupItemsState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, groupItemsState.props));
    $$renderer2.push(`<div style="display: contents;">`);
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { ref });
  });
}
function Command_input($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      value = "",
      autofocus = false,
      id = createId(uid),
      ref = null,
      child,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const inputState = CommandInputState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      value: boxWith(() => value, (v2) => {
        value = v2;
      }),
      autofocus: boxWith(() => autofocus ?? false)
    });
    const mergedProps = derived(() => mergeProps(restProps, inputState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<input${attributes({ ...mergedProps(), value }, void 0, void 0, void 0, 4)}/>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { value, ref });
  });
}
function Command_item($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      value = "",
      disabled = false,
      children,
      child,
      onSelect = noop,
      forceMount = false,
      keywords = [],
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const itemState = CommandItemState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      value: boxWith(() => value),
      disabled: boxWith(() => disabled),
      onSelect: boxWith(() => onSelect),
      forceMount: boxWith(() => forceMount),
      keywords: boxWith(() => keywords)
    });
    const mergedProps = derived(() => mergeProps(restProps, itemState.props));
    $$renderer2.push(`<!---->`);
    {
      $$renderer2.push(`<div style="display: contents;" data-item-wrapper=""${attr("data-value", itemState.trueValue)}>`);
      if (itemState.shouldRender) {
        $$renderer2.push("<!--[0-->");
        if (child) {
          $$renderer2.push("<!--[0-->");
          child($$renderer2, { props: mergedProps() });
          $$renderer2.push(`<!---->`);
        } else {
          $$renderer2.push("<!--[-1-->");
          $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
          children?.($$renderer2);
          $$renderer2.push(`<!----></div>`);
        }
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!---->`);
    bind_props($$props, { ref });
  });
}
function Command_list($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      child,
      children,
      "aria-label": ariaLabel,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const listState = CommandListState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      ariaLabel: boxWith(() => ariaLabel ?? "Suggestions...")
    });
    const mergedProps = derived(() => mergeProps(restProps, listState.props));
    $$renderer2.push(`<!---->`);
    {
      if (child) {
        $$renderer2.push("<!--[0-->");
        child($$renderer2, { props: mergedProps() });
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
        children?.($$renderer2);
        $$renderer2.push(`<!----></div>`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!---->`);
    bind_props($$props, { ref });
  });
}
function Command_viewport($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      children,
      child,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const listViewportState = CommandViewportState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, listViewportState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Command_separator($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      ref = null,
      forceMount = false,
      children,
      child,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const separatorState = CommandSeparatorState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      forceMount: boxWith(() => forceMount)
    });
    const mergedProps = derived(() => mergeProps(restProps, separatorState.props));
    if (separatorState.shouldRender) {
      $$renderer2.push("<!--[0-->");
      if (child) {
        $$renderer2.push("<!--[0-->");
        child($$renderer2, { props: mergedProps() });
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
        children?.($$renderer2);
        $$renderer2.push(`<!----></div>`);
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
const SCORE_CONTINUE_MATCH = 1;
const SCORE_SPACE_WORD_JUMP = 0.9;
const SCORE_NON_SPACE_WORD_JUMP = 0.8;
const SCORE_CHARACTER_JUMP = 0.17;
const SCORE_TRANSPOSITION = 0.1;
const PENALTY_SKIPPED = 0.999;
const PENALTY_CASE_MISMATCH = 0.9999;
const PENALTY_NOT_COMPLETE = 0.99;
const IS_GAP_REGEXP = /[\\/_+.#"@[({&]/;
const COUNT_GAPS_REGEXP = /[\\/_+.#"@[({&]/g;
const IS_SPACE_REGEXP = /[\s-]/;
const COUNT_SPACE_REGEXP = /[\s-]/g;
function computeCommandScoreInner(string, abbreviation, lowerString, lowerAbbreviation, stringIndex, abbreviationIndex, memoizedResults) {
  if (abbreviationIndex === abbreviation.length) {
    if (stringIndex === string.length)
      return SCORE_CONTINUE_MATCH;
    return PENALTY_NOT_COMPLETE;
  }
  const memoizeKey = `${stringIndex},${abbreviationIndex}`;
  if (memoizedResults[memoizeKey] !== void 0)
    return memoizedResults[memoizeKey];
  const abbreviationChar = lowerAbbreviation.charAt(abbreviationIndex);
  let index = lowerString.indexOf(abbreviationChar, stringIndex);
  let highScore = 0;
  let score, transposedScore, wordBreaks, spaceBreaks;
  while (index >= 0) {
    score = computeCommandScoreInner(string, abbreviation, lowerString, lowerAbbreviation, index + 1, abbreviationIndex + 1, memoizedResults);
    if (score > highScore) {
      if (index === stringIndex) {
        score *= SCORE_CONTINUE_MATCH;
      } else if (IS_GAP_REGEXP.test(string.charAt(index - 1))) {
        score *= SCORE_NON_SPACE_WORD_JUMP;
        wordBreaks = string.slice(stringIndex, index - 1).match(COUNT_GAPS_REGEXP);
        if (wordBreaks && stringIndex > 0) {
          score *= PENALTY_SKIPPED ** wordBreaks.length;
        }
      } else if (IS_SPACE_REGEXP.test(string.charAt(index - 1))) {
        score *= SCORE_SPACE_WORD_JUMP;
        spaceBreaks = string.slice(stringIndex, index - 1).match(COUNT_SPACE_REGEXP);
        if (spaceBreaks && stringIndex > 0) {
          score *= PENALTY_SKIPPED ** spaceBreaks.length;
        }
      } else {
        score *= SCORE_CHARACTER_JUMP;
        if (stringIndex > 0) {
          score *= PENALTY_SKIPPED ** (index - stringIndex);
        }
      }
      if (string.charAt(index) !== abbreviation.charAt(abbreviationIndex)) {
        score *= PENALTY_CASE_MISMATCH;
      }
    }
    if (score < SCORE_TRANSPOSITION && lowerString.charAt(index - 1) === lowerAbbreviation.charAt(abbreviationIndex + 1) || lowerAbbreviation.charAt(abbreviationIndex + 1) === lowerAbbreviation.charAt(abbreviationIndex) && lowerString.charAt(index - 1) !== lowerAbbreviation.charAt(abbreviationIndex)) {
      transposedScore = computeCommandScoreInner(string, abbreviation, lowerString, lowerAbbreviation, index + 1, abbreviationIndex + 2, memoizedResults);
      if (transposedScore * SCORE_TRANSPOSITION > score) {
        score = transposedScore * SCORE_TRANSPOSITION;
      }
    }
    if (score > highScore) {
      highScore = score;
    }
    index = lowerString.indexOf(abbreviationChar, index + 1);
  }
  memoizedResults[memoizeKey] = highScore;
  return highScore;
}
function formatInput(string) {
  return string.toLowerCase().replace(COUNT_SPACE_REGEXP, " ");
}
function computeCommandScore(command, search2, commandKeywords) {
  command = commandKeywords && commandKeywords.length > 0 ? `${`${command} ${commandKeywords?.join(" ")}`}` : command;
  return computeCommandScoreInner(command, search2, formatInput(command), formatInput(search2), 0, 0, {});
}
function Dialog($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      open = false,
      onOpenChange = noop,
      onOpenChangeComplete = noop,
      children
    } = $$props;
    DialogRootState.create({
      variant: boxWith(() => "dialog"),
      open: boxWith(() => open, (v2) => {
        open = v2;
        onOpenChange(v2);
      }),
      onOpenChangeComplete: boxWith(() => onOpenChangeComplete)
    });
    children?.($$renderer2);
    $$renderer2.push(`<!---->`);
    bind_props($$props, { open });
  });
}
function Dialog_content($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      children,
      child,
      ref = null,
      forceMount = false,
      onCloseAutoFocus = noop,
      onOpenAutoFocus = noop,
      onEscapeKeydown = noop,
      onInteractOutside = noop,
      trapFocus = true,
      preventScroll = true,
      restoreScrollDelay = null,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const contentState = DialogContentState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, contentState.props));
    if (contentState.shouldRender || forceMount) {
      $$renderer2.push("<!--[0-->");
      {
        let focusScope = function($$renderer3, { props: focusScopeProps }) {
          Escape_layer($$renderer3, spread_props([
            mergedProps(),
            {
              enabled: contentState.root.opts.open.current,
              ref: contentState.opts.ref,
              onEscapeKeydown: (e) => {
                onEscapeKeydown(e);
                if (e.defaultPrevented) return;
                contentState.root.handleClose();
              },
              children: ($$renderer4) => {
                Dismissible_layer($$renderer4, spread_props([
                  mergedProps(),
                  {
                    ref: contentState.opts.ref,
                    enabled: contentState.root.opts.open.current,
                    onInteractOutside: (e) => {
                      onInteractOutside(e);
                      if (e.defaultPrevented) return;
                      contentState.root.handleClose();
                    },
                    children: ($$renderer5) => {
                      Text_selection_layer($$renderer5, spread_props([
                        mergedProps(),
                        {
                          ref: contentState.opts.ref,
                          enabled: contentState.root.opts.open.current,
                          children: ($$renderer6) => {
                            if (child) {
                              $$renderer6.push("<!--[0-->");
                              if (contentState.root.opts.open.current) {
                                $$renderer6.push("<!--[0-->");
                                Scroll_lock($$renderer6, { preventScroll, restoreScrollDelay });
                              } else {
                                $$renderer6.push("<!--[-1-->");
                              }
                              $$renderer6.push(`<!--]--> `);
                              child($$renderer6, {
                                props: mergeProps(mergedProps(), focusScopeProps),
                                ...contentState.snippetProps
                              });
                              $$renderer6.push(`<!---->`);
                            } else {
                              $$renderer6.push("<!--[-1-->");
                              Scroll_lock($$renderer6, { preventScroll });
                              $$renderer6.push(`<!----> <div${attributes({ ...mergeProps(mergedProps(), focusScopeProps) })}>`);
                              children?.($$renderer6);
                              $$renderer6.push(`<!----></div>`);
                            }
                            $$renderer6.push(`<!--]-->`);
                          },
                          $$slots: { default: true }
                        }
                      ]));
                    },
                    $$slots: { default: true }
                  }
                ]));
              },
              $$slots: { default: true }
            }
          ]));
        };
        Focus_scope($$renderer2, {
          ref: contentState.opts.ref,
          loop: true,
          trapFocus,
          enabled: contentState.root.opts.open.current,
          onOpenAutoFocus,
          onCloseAutoFocus,
          focusScope
        });
      }
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function useColorMode() {
  let isDark = DEV;
  return {
    get current() {
      return "light";
    },
    get isDark() {
      return isDark;
    }
  };
}
function stableFilters(filters) {
  return Object.entries(filters).filter(([, value]) => value !== "").sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0);
}
function nodeFingerprint(id, config, edges) {
  const preds = edges.filter((e) => e.target === id).map((e) => `${e.source}#${e.targetHandle ?? ""}`).sort();
  return JSON.stringify({
    preds,
    enabled: config.enabled,
    q: config.q,
    // A File can't be serialized — name+size+mtime identifies the upload.
    image: config.image ? `${config.image.name}:${config.image.size}:${config.image.lastModified}` : null,
    where: config.where,
    filters: stableFilters(config.filters),
    mode: config.mode,
    n: config.n,
    rerank: config.rerank,
    minScore: config.minScore,
    refineScope: config.refineScope,
    combineMode: config.combineMode,
    tags: config.tags,
    atlas: config.capturedAtlasSelection ? config.capturedAtlasSelection.map(hitKey) : null
  });
}
class WorkflowTags {
  /** tags keyed by `hitKey` (the dataset's identity key). */
  byChunk = {};
  /** Current tags for a chunk (empty array when untagged). */
  forHit(hit) {
    return this.byChunk[hitKey(hit)] ?? [];
  }
  /** Toggle one tag on one chunk (inline tagging from a results row). */
  toggle(hit, tag) {
    const t = tag.trim();
    if (!t) return;
    const k2 = hitKey(hit);
    const cur = this.byChunk[k2] ?? [];
    const next = cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t];
    this.byChunk = { ...this.byChunk, [k2]: next };
  }
  /** Stamp `tags` (union, add-only) onto every chunk — the Tagger node's bulk
   *  write and the inline add path. */
  addTo(hits, tags) {
    const add = tags.map((t) => t.trim()).filter(Boolean);
    if (!add.length || !hits.length) return;
    const next = { ...this.byChunk };
    for (const h2 of hits) {
      const k2 = hitKey(h2);
      const cur = next[k2] ?? [];
      next[k2] = [...cur, ...add.filter((t) => !cur.includes(t))];
    }
    this.byChunk = next;
  }
  /** Hits with their current tags attached from the store — used at export time
   *  so a `tags` column reflects both inline and Tagger-node tags. */
  withTags(hits) {
    return hits.map((h2) => ({ ...h2, tags: this.forHit(h2) }));
  }
  /** Replace the whole store (rehydrate from persistence). */
  hydrate(byChunk) {
    this.byChunk = byChunk;
  }
  /** A plain serialisable snapshot of the store (for persistence). */
  snapshot() {
    return { ...this.byChunk };
  }
  /** Clear all tags (full graph reset). */
  reset() {
    this.byChunk = {};
  }
}
class UndoHistory {
  undoStack = [];
  redoStack = [];
  limit;
  constructor(limit = 50) {
    this.limit = limit;
  }
  get canUndo() {
    return this.undoStack.length > 0;
  }
  get canRedo() {
    return this.redoStack.length > 0;
  }
  /** Record `snapshot` (the pre-change state) and invalidate redo. */
  push(snapshot2) {
    this.undoStack = [...this.undoStack, snapshot2].slice(-this.limit);
    this.redoStack = [];
  }
  /** Pop the undo stack; stashes `current` for redo. Returns the snapshot to
   *  restore, or null when there is nothing to undo. */
  undo(current) {
    const prev = this.undoStack.at(-1);
    if (prev === void 0) return null;
    this.redoStack = [...this.redoStack, current].slice(-this.limit);
    this.undoStack = this.undoStack.slice(0, -1);
    return prev;
  }
  /** Pop the redo stack; stashes `current` for undo. Returns the snapshot to
   *  restore, or null when there is nothing to redo. */
  redo(current) {
    const next = this.redoStack.at(-1);
    if (next === void 0) return null;
    this.undoStack = [...this.undoStack, current].slice(-this.limit);
    this.redoStack = this.redoStack.slice(0, -1);
    return next;
  }
  /** Drop all history (full graph reset). */
  clear() {
    this.undoStack = [];
    this.redoStack = [];
  }
}
const elk = new ELK();
const DEFAULT_W = 256;
const DEFAULT_H = 150;
async function autoLayout(nodes, edges, direction = "LR") {
  if (!nodes.length) return nodes;
  const size = (n2) => ({
    width: n2.measured?.width ?? n2.width ?? DEFAULT_W,
    height: n2.measured?.height ?? n2.height ?? DEFAULT_H
  });
  const nodeIds = new Set(nodes.map((n2) => n2.id));
  const graph2 = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": direction === "LR" ? "RIGHT" : "DOWN",
      "elk.layered.spacing.nodeNodeBetweenLayers": "90",
      // gap between ranks
      "elk.spacing.nodeNode": "40",
      // gap between siblings
      "elk.edgeRouting": "ORTHOGONAL",
      // clean right-angle edges (dagre couldn't)
      "elk.padding": "[top=20,left=20,bottom=20,right=20]"
    },
    children: nodes.map((n2) => ({ id: n2.id, ...size(n2) })),
    edges: edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target)).map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] }))
  };
  const laid = await elk.layout(graph2);
  const positioned = new Map((laid.children ?? []).map((c) => [c.id, c]));
  return nodes.map((n2) => {
    const p2 = positioned.get(n2.id);
    return p2 && p2.x != null && p2.y != null ? { ...n2, position: { x: p2.x, y: p2.y } } : n2;
  });
}
const MAX_SCOPE_DOCS = 80;
const MAX_SCOPE_CHUNKS = 300;
function sqlQuote(value) {
  return `'${value.replace(/'/g, "''")}'`;
}
function dedupeHits(hits) {
  const seen = /* @__PURE__ */ new Set();
  return hits.filter((h2) => {
    const k2 = hitKey(h2);
    if (seen.has(k2)) return false;
    seen.add(k2);
    return true;
  });
}
function keyLiteral(value) {
  return typeof value === "number" ? String(value) : sqlQuote(String(value ?? ""));
}
function videoScopeClause(hits) {
  const view = activeView();
  const docKey = view.docKeyField;
  const all = [...new Set(hits.map((h2) => String(h2[docKey] ?? "")))];
  const docs = all.slice(0, MAX_SCOPE_DOCS);
  if (!docs.length) return null;
  return {
    clause: `${docKey} IN (${docs.map(sqlQuote).join(", ")})`,
    count: docs.length,
    capped: all.length > docs.length
  };
}
function chunkScopeClause(hits) {
  const view = activeView();
  const keyFields = view.keyFields;
  const uniq = dedupeHits(hits);
  const picked = uniq.slice(0, MAX_SCOPE_CHUNKS);
  if (!picked.length) return null;
  const terms = picked.map((h2) => {
    const row = h2;
    const conds = keyFields.map((k2) => `${k2} = ${keyLiteral(row[k2])}`);
    return `(${conds.join(" AND ")})`;
  });
  return {
    clause: `(${terms.join(" OR ")})`,
    count: picked.length,
    capped: uniq.length > picked.length
  };
}
const NODE_KINDS = [
  "query",
  "image",
  "filter",
  "atlas",
  "search",
  "combine",
  "tagger",
  "results",
  "export"
];
function isNodeKind(v2) {
  return v2 !== void 0 && NODE_KINDS.includes(v2);
}
const RERANK_TOP_N = 20;
const SEARCH_IN_HANDLE = "in";
const SEARCH_IMAGE_HANDLE = "image";
const DEFAULT_N = 24;
const MIN_N = 1;
const MAX_N = 100;
async function runGraph(deps) {
  const ids = deps.nodes.map((n2) => n2.id);
  const incoming = incomingMap(deps.edges, new Set(ids));
  const order = topoOrder(ids, incoming);
  if (!order) return CYCLE_ERROR;
  const outputs = /* @__PURE__ */ new Map();
  for (const id of order) {
    outputs.set(id, computeNode(deps, id, incoming.get(id) ?? [], outputs));
  }
  await Promise.all(outputs.values());
  return null;
}
async function runSubgraph(deps, targetId, opts = {}) {
  const ids = new Set(deps.nodes.map((n2) => n2.id));
  if (!ids.has(targetId)) return { error: null, ran: [] };
  const incoming = incomingMap(deps.edges, ids);
  const closure = /* @__PURE__ */ new Set([targetId]);
  const stack = [targetId];
  while (stack.length) {
    const id = stack.pop();
    for (const p2 of incoming.get(id) ?? []) {
      if (!closure.has(p2)) {
        closure.add(p2);
        stack.push(p2);
      }
    }
  }
  const order = topoOrder([...closure], incoming);
  if (!order) return { error: CYCLE_ERROR, ran: [] };
  const outputs = /* @__PURE__ */ new Map();
  const ran = [];
  const ranSet = /* @__PURE__ */ new Set();
  for (const id of order) {
    const preds = incoming.get(id) ?? [];
    const predRecomputed = preds.some((p2) => ranSet.has(p2));
    const cached = id === targetId || opts.fresh || predRecomputed ? null : deps.cachedOutput(id);
    if (cached) {
      outputs.set(id, Promise.resolve(cached));
      continue;
    }
    ran.push(id);
    ranSet.add(id);
    outputs.set(id, computeNode(deps, id, preds, outputs));
  }
  await Promise.all(outputs.values());
  return { error: null, ran };
}
const CYCLE_ERROR = "The graph has a cycle — remove a connection and run again.";
function incomingMap(edges, ids) {
  const incoming = new Map([...ids].map((id) => [id, []]));
  for (const e of edges) {
    if (incoming.has(e.target) && ids.has(e.source)) incoming.get(e.target).push(e.source);
  }
  return incoming;
}
function computeNode(deps, id, preds, outputs) {
  return Promise.all(preds.map((p2) => outputs.get(p2))).then(async (predOutputs) => {
    const key2 = deps.fingerprint(id);
    const out = await runNode(deps, id, predOutputs);
    deps.patchRuntime(id, { output: out, outputKey: key2, stale: false });
    return out;
  });
}
function topoOrder(ids, incoming) {
  const members = new Set(ids);
  const deg = new Map(
    ids.map((id) => [id, (incoming.get(id) ?? []).filter((p2) => members.has(p2)).length])
  );
  const outgoing = /* @__PURE__ */ new Map();
  for (const id of ids) {
    for (const p2 of incoming.get(id) ?? []) {
      if (!members.has(p2)) continue;
      const list = outgoing.get(p2);
      if (list) list.push(id);
      else outgoing.set(p2, [id]);
    }
  }
  const queue = ids.filter((id) => (deg.get(id) ?? 0) === 0);
  const order = [];
  while (queue.length) {
    const id = queue.shift();
    order.push(id);
    for (const t of outgoing.get(id) ?? []) {
      const d = (deg.get(t) ?? 0) - 1;
      deg.set(t, d);
      if (d === 0) queue.push(t);
    }
  }
  return order.length === ids.length ? order : null;
}
async function runNode(deps, id, predOutputs) {
  const kind = deps.kindOf(id);
  if (kind === null) return { spec: {}, hits: null };
  const cfg = deps.config(id);
  if (predOutputs.some((o) => o.failed)) {
    deps.patchRuntime(id, { status: "error", error: "Skipped — an upstream node failed." });
    return { spec: {}, hits: null, failed: true };
  }
  const inSpec = {};
  const scopeHits = [];
  const sourceHitSets = [];
  let qContrib = 0;
  let imgContrib = 0;
  for (const o of predOutputs) {
    if (o.spec.q) qContrib += 1;
    if (o.spec.image) imgContrib += 1;
    const { filters, ...rest } = o.spec;
    Object.assign(inSpec, rest);
    if (filters) inSpec.filters = { ...inSpec.filters, ...filters };
    if (o.hits && o.hits.length) {
      scopeHits.push(...o.hits);
      sourceHitSets.push(o.hits);
    }
  }
  const scope = scopeHits.length ? dedupeHits(scopeHits) : null;
  if (!cfg.enabled) {
    deps.patchRuntime(id, { status: "idle", hits: scope, count: scope?.length ?? null });
    return { spec: {}, hits: scope };
  }
  try {
    switch (kind) {
      case "query": {
        const q = cfg.q.trim();
        deps.patchRuntime(id, { status: q ? "done" : "idle" });
        return { spec: q ? { q } : {}, hits: null };
      }
      case "image": {
        deps.patchRuntime(id, { status: cfg.image ? "done" : "idle" });
        return { spec: cfg.image ? { image: cfg.image } : {}, hits: null };
      }
      case "filter": {
        const spec = {};
        if (cfg.where.trim()) spec.where = cfg.where.trim();
        const filters = {};
        for (const [field, value] of Object.entries(cfg.filters)) {
          const v2 = value.trim();
          if (v2) filters[field] = v2;
        }
        if (Object.keys(filters).length) spec.filters = filters;
        deps.patchRuntime(id, { status: Object.keys(spec).length ? "done" : "idle" });
        return { spec, hits: null };
      }
      case "atlas": {
        const captured = cfg.capturedAtlasSelection;
        const hits = captured && captured.length ? captured : null;
        deps.patchRuntime(id, {
          status: hits ? "done" : "idle",
          hits,
          count: hits?.length ?? null
        });
        return { spec: {}, hits };
      }
      case "combine": {
        let combined = [];
        if (sourceHitSets.length) {
          if (cfg.combineMode === "intersect") {
            const keySets = sourceHitSets.map((s) => new Set(s.map(hitKey)));
            combined = dedupeHits(
              sourceHitSets[0].filter((h2) => keySets.every((ks) => ks.has(hitKey(h2))))
            );
          } else {
            combined = scope ?? [];
          }
        }
        deps.patchRuntime(id, {
          status: sourceHitSets.length ? "done" : "idle",
          hits: combined,
          count: combined.length
        });
        return { spec: {}, hits: combined.length ? combined : null };
      }
      case "tagger": {
        if (scope) deps.tagHits(scope, cfg.tags);
        deps.patchRuntime(id, {
          status: scope ? "done" : "idle",
          hits: scope,
          count: scope?.length ?? null
        });
        return { spec: {}, hits: scope };
      }
      case "search": {
        const q = inSpec.q?.trim() || cfg.q.trim();
        const image = inSpec.image ?? null;
        const inlineQDropped = cfg.q.trim() && qContrib > 0 ? 1 : 0;
        const droppedInputs = Math.max(0, qContrib - 1) + Math.max(0, imgContrib - 1) + inlineQDropped;
        if (!q && !image) {
          deps.patchRuntime(id, {
            status: "idle",
            hits: scope,
            count: scope?.length ?? null,
            droppedInputs
          });
          return { spec: {}, hits: scope };
        }
        deps.patchRuntime(id, { status: "running" });
        const spec = { q, n: cfg.n, mode: cfg.mode };
        if (cfg.rerank) {
          spec.rerank = true;
          spec.rerankN = RERANK_TOP_N;
        }
        if (image) spec.image = image;
        if (inSpec.filters) spec.filters = inSpec.filters;
        const wheres = [];
        if (inSpec.where) wheres.push(inSpec.where);
        let scopedDocs = null;
        let scopedChunks = null;
        let scopeCapped = false;
        if (scope?.length) {
          const sc = cfg.refineScope === "chunk" ? chunkScopeClause(scope) : videoScopeClause(scope);
          if (sc) {
            wheres.push(sc.clause);
            scopeCapped = sc.capped;
            if (cfg.refineScope === "chunk") scopedChunks = sc.count;
            else scopedDocs = sc.count;
          }
        }
        if (wheres.length) spec.where = wheres.map((w) => `(${w})`).join(" AND ");
        const t0 = performance.now();
        let hits = await search(spec);
        const ms = Math.round(performance.now() - t0);
        if (cfg.minScore != null) {
          const min = cfg.minScore;
          hits = hits.filter((h2) => {
            const r = relevanceOf(h2);
            return r === null || r >= min;
          });
        }
        deps.patchRuntime(id, {
          status: "done",
          hits,
          count: hits.length,
          ms,
          scopedDocs,
          scopedChunks,
          scopeCapped,
          droppedInputs
        });
        return { spec: {}, hits };
      }
      // Sinks: collect the incoming hits and surface them (Results renders them;
      // Export downloads them). Neither contributes a spec.
      case "results":
      case "export": {
        deps.patchRuntime(id, {
          status: scope ? "done" : "idle",
          hits: scope,
          count: scope?.length ?? null
        });
        return { spec: {}, hits: scope };
      }
      default: {
        const _exhaustive = kind;
        return _exhaustive;
      }
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    deps.patchRuntime(id, { status: "error", error: msg });
    return { spec: {}, hits: null, failed: true };
  }
}
const MODE_VALUES = [
  "fts",
  "semantic",
  "visual",
  "scene",
  "scene_fts",
  "hybrid",
  "all"
];
const SearchModeSchema = v.picklist(MODE_VALUES);
const ConfigSchema = v.object({
  q: v.fallback(v.string(), ""),
  imageName: v.fallback(v.string(), ""),
  where: v.fallback(v.string(), ""),
  filters: v.fallback(v.record(v.string(), v.string()), () => ({})),
  mode: v.fallback(SearchModeSchema, "fts"),
  n: v.pipe(
    v.fallback(v.number(), DEFAULT_N),
    v.transform((n2) => Math.max(MIN_N, Math.min(MAX_N, Math.round(n2))))
  ),
  rerank: v.fallback(v.boolean(), false),
  minScore: v.fallback(v.nullable(v.number()), null),
  refineScope: v.fallback(v.picklist(["video", "chunk"]), "video"),
  combineMode: v.fallback(v.picklist(["union", "intersect"]), "union"),
  tags: v.fallback(v.array(v.string()), () => []),
  exportFormat: v.fallback(v.picklist(["json", "csv"]), "csv"),
  // `null` = every column the active dataset offers; stale field names in a
  // persisted array self-heal at export time (orderColumns drops unknowns).
  exportColumns: v.fallback(v.nullable(v.array(v.string())), null),
  // Atlas modal capture is a full Row[] (media path, alignments, …) — too heavy
  // and stale-prone to round-trip through localStorage, so it is NOT persisted:
  // the schema always heals it to null, and a reload discards the capture (the
  // user re-opens the modal to re-select). TODO: persist a minimal identity-key
  // set + rehydrate via /api/atlas/chunks if needed.
  capturedAtlasSelection: v.fallback(v.null_(), null),
  label: v.fallback(v.string(), ""),
  enabled: v.fallback(v.boolean(), true)
});
const PersistedNodeSchema = v.object({
  id: v.string(),
  type: v.picklist(NODE_KINDS),
  position: v.object({ x: v.number(), y: v.number() })
});
const PersistedEdgeSchema = v.object({
  id: v.string(),
  source: v.string(),
  target: v.string(),
  // Which target port the edge lands on (Search has "in" + "image"); absent in
  // pre-two-port snapshots — the loader infers it from the source kind.
  targetHandle: v.optional(v.string()),
  label: v.optional(v.string())
});
const PersistedGraphSchema = v.object({
  nodes: v.pipe(v.array(PersistedNodeSchema), v.minLength(1)),
  edges: v.optional(v.array(PersistedEdgeSchema), []),
  config: v.optional(v.record(v.string(), ConfigSchema), {}),
  tags: v.optional(v.record(v.string(), v.array(v.string())), {})
});
function safeParseGraph(raw) {
  if (!raw) return null;
  try {
    const result = v.safeParse(PersistedGraphSchema, JSON.parse(raw));
    return result.success ? result.output : null;
  } catch {
    return null;
  }
}
const HISTORY_LIMIT = 50;
const DUPLICATE_OFFSET_PX = 48;
const PASTE_OFFSET_PX = 32;
const SEARCH_MODES = [
  { value: "fts", label: "Keyword (FTS)" },
  { value: "semantic", label: "Meaning (vector)" },
  { value: "hybrid", label: "Hybrid (FTS + vector)" },
  { value: "visual", label: "Image (frame vector)" },
  { value: "scene", label: "Scene (caption vector)" },
  { value: "scene_fts", label: "Scene (caption keyword)" },
  { value: "all", label: "All judges (RRF)" }
];
const modeLabel = (mode) => SEARCH_MODES.find((m) => m.value === mode)?.label ?? mode;
const KIND_LABEL = {
  query: "Text query",
  image: "Image",
  filter: "Filter",
  atlas: "Atlas selection",
  search: "Search",
  combine: "Combine",
  tagger: "Tagger",
  results: "Results",
  export: "Export"
};
const nodeLabel = (kind) => KIND_LABEL[kind];
const STATUS_DOT = {
  idle: "bg-muted-foreground/40",
  running: "bg-primary animate-pulse",
  done: "bg-emerald-500",
  error: "bg-destructive"
};
function defaultConfig() {
  return {
    q: "",
    image: null,
    imageName: "",
    where: "",
    filters: {},
    mode: "fts",
    n: DEFAULT_N,
    rerank: false,
    minScore: null,
    refineScope: "video",
    combineMode: "union",
    tags: [],
    exportFormat: "csv",
    exportColumns: null,
    capturedAtlasSelection: null,
    label: "",
    enabled: true
  };
}
function blankRuntime() {
  return {
    status: "idle",
    error: null,
    hits: null,
    count: null,
    ms: null,
    scopedDocs: null,
    scopedChunks: null,
    scopeCapped: false,
    droppedInputs: 0,
    output: null,
    outputKey: null,
    stale: false
  };
}
class WorkflowGraph {
  /** The Svelte Flow nodes — bound into <SvelteFlow bind:nodes>. */
  nodes = [];
  /** The Svelte Flow edges — bound into <SvelteFlow bind:edges>. */
  edges = [];
  /** Per-node user input, keyed by node id. Deep `$state` ON PURPOSE: node
   *  components mutate fields in place (`bind:value={cfg.q}`,
   *  `bind:checked={cfg.rerank}`, the Inspector's label bind), and the autosave
   *  `$effect` tracks those deep writes via `snapshot()`. Must NOT become
   *  `$state.raw` — that would silently de-reactify every such bind. */
  config = {};
  /** Per-node run state, keyed by node id. */
  runtime = {};
  /** Shared tag store (chunk-identity keyed) — inline tagging in the results list
   *  and the Tagger node both write here; tags persist + flow into Export. */
  tags = new WorkflowTags();
  /** True while `run()` is in flight (disables the Run button, shows spinner). */
  running = false;
  /** Last graph-level failure (cycle, etc.); per-node errors live in `runtime`. */
  lastError = null;
  /** A result the user clicked — plays in the Inspector. */
  selectedHit = null;
  /** The node whose intermediate state the Inspector shows (click a node). */
  inspectedNodeId = null;
  /** Ids currently selected on the canvas (tracked via `onselectionchange`) —
   *  drives the toolbar's Delete button. */
  selectedNodeIds = [];
  selectedEdgeIds = [];
  get hasSelection() {
    return this.selectedNodeIds.length > 0 || this.selectedEdgeIds.length > 0;
  }
  /** Undo/redo stacks (snapshot strings) — its own focused store. */
  undoHistory = new UndoHistory(HISTORY_LIMIT);
  /** Last snapshot pushed to history, so the debounced checkpoint can diff. */
  lastCheckpoint = "";
  /** Copy/paste buffer of detached nodes + a paste counter (cascade offset). */
  clipboard = [];
  pasteCount = 0;
  get canUndo() {
    return this.undoHistory.canUndo;
  }
  get canRedo() {
    return this.undoHistory.canRedo;
  }
  /** Monotonic id source for nodes added at runtime (seeds use bare-kind ids). */
  seq = 0;
  constructor() {
    if (!this.load()) this.seed();
  }
  /** Kind of a node by id (its Svelte Flow `type`). */
  kindOf(id) {
    const t = this.nodes.find((x) => x.id === id)?.type;
    return isNodeKind(t) ? t : null;
  }
  /**
   * The starter graph: a multi-modal refinement chain that demonstrates how
   * to wire "multiples". Upload an image and Run to see all four stages; with
   * no image the visual stage politely skips and the Scene→Keyword chain still
   * runs, so the refinement is visible either way.
   *
   *   Image ─image─▶ Search·Visual ─refine─▶ Search·Scene ─refine─▶ Search·Keyword ─▶ Results
   */
  seed() {
    const query = (mode, q) => ({ ...defaultConfig(), mode, q });
    this.config = {
      image: defaultConfig(),
      "search-visual": query("visual", ""),
      "search-scene": query("scene", "talarstol"),
      "search-said": query("fts", "skatt"),
      results: defaultConfig()
    };
    this.runtime = Object.fromEntries(Object.keys(this.config).map((id) => [id, blankRuntime()]));
    this.tags.reset();
    this.undoHistory.clear();
    this.nodes = [
      {
        id: "image",
        type: "image",
        position: { x: -60, y: 60 },
        data: {}
      },
      {
        id: "search-visual",
        type: "search",
        position: { x: 240, y: 40 },
        data: {}
      },
      {
        id: "search-scene",
        type: "search",
        position: { x: 560, y: 100 },
        data: {}
      },
      {
        id: "search-said",
        type: "search",
        position: { x: 880, y: 160 },
        data: {}
      },
      {
        id: "results",
        type: "results",
        position: { x: 1200, y: 100 },
        data: {}
      }
    ];
    this.edges = [
      {
        id: "e-img",
        source: "image",
        target: "search-visual",
        targetHandle: SEARCH_IMAGE_HANDLE,
        label: "image"
      },
      {
        id: "e-v-scene",
        source: "search-visual",
        target: "search-scene",
        targetHandle: SEARCH_IN_HANDLE,
        label: "refine"
      },
      {
        id: "e-scene-said",
        source: "search-scene",
        target: "search-said",
        targetHandle: SEARCH_IN_HANDLE,
        label: "refine"
      },
      { id: "e-said-res", source: "search-said", target: "results" }
    ];
    this.seq = 0;
    this.running = false;
    this.lastError = null;
    this.selectedHit = null;
    this.inspectedNodeId = null;
    this.selectedNodeIds = [];
    this.selectedEdgeIds = [];
    this.pasteCount = 0;
    this.lastCheckpoint = this.snapshot();
  }
  /** Add a fresh node of `kind` at `position`; returns its id. */
  addNode(kind, position) {
    if (this.running) return "";
    const id = `${kind}-${++this.seq}`;
    this.config = { ...this.config, [id]: defaultConfig() };
    this.runtime = { ...this.runtime, [id]: blankRuntime() };
    this.nodes = [...this.nodes, { id, type: kind, position, data: {} }];
    return id;
  }
  /** Duplicate a node (its config, sans the image File) at a small offset. */
  duplicateNode(id) {
    const src = this.config[id];
    const kind = this.kindOf(id);
    if (!src || !kind || this.running) return id;
    const newId = `${kind}-${++this.seq}`;
    const srcNode = this.nodes.find((n2) => n2.id === id);
    const pos = srcNode ? {
      x: srcNode.position.x + DUPLICATE_OFFSET_PX,
      y: srcNode.position.y + DUPLICATE_OFFSET_PX
    } : { x: 0, y: 0 };
    this.config = {
      ...this.config,
      [newId]: {
        ...src,
        image: null,
        label: src.label ? `${src.label} copy` : ""
      }
    };
    this.runtime = { ...this.runtime, [newId]: blankRuntime() };
    this.nodes = [
      ...this.nodes,
      { id: newId, type: kind, position: pos, data: {} }
    ];
    return newId;
  }
  /** Patch one node's user input. `config` is deep `$state`, so components'
   *  in-place `bind:` mutations are equally reactive — this helper exists for
   *  multi-field patches and call-site ergonomics (falls back to
   *  `defaultConfig()` when the id has no config yet). */
  setConfig(id, patch) {
    const prev = this.config[id] ?? defaultConfig();
    this.config = { ...this.config, [id]: { ...prev, ...patch } };
  }
  /** Current fingerprint of a node's output-affecting config + incoming
   *  edges (see fingerprint.ts) — a mismatch with the stored `outputKey`
   *  means the node was edited or rewired since its output was recorded. */
  nodeFingerprint(id) {
    return nodeFingerprint(id, this.config[id] ?? defaultConfig(), this.edges);
  }
  /** True when a node's last results were computed from a different config or
   *  wiring than the current one — drives the "stale" badge live (config is
   *  deep $state and edges are $state, so reads here re-derive on edit). */
  isOutdated(id) {
    const rt = this.runtime[id];
    if (!rt?.output || rt.outputKey === null) return false;
    return rt.outputKey !== this.nodeFingerprint(id);
  }
  // ── Connection validation (keeps the graph a DAG) ───────────────────────────
  /** Validate a would-be edge (wired to `<SvelteFlow isValidConnection>`): no
   *  self-loop, no duplicate edge, and no edge that would create a cycle (the
   *  target must not already reach the source). Port direction is enforced by
   *  the node Handles (sinks expose no source port; sources no target port). */
  canConnect(connection) {
    return this.connectionError(connection) === null;
  }
  /** Why a would-be connection is invalid (for user feedback), or null if it is
   *  valid — also the source of truth for `canConnect` and `isValidConnection`
   *  (which gates edge reconnection too). */
  connectionError(connection) {
    const { source, target } = connection;
    if (!source || !target) return null;
    if (source === target) return "A node can't connect to itself";
    if (this.edges.some((e) => e.source === source && e.target === target)) return "Those nodes are already connected";
    if (this.reaches(this.edges, target, source)) return "That would create a loop";
    const targetHandle = connection.targetHandle ?? null;
    const sourceKind = this.kindOf(source);
    if (targetHandle === SEARCH_IMAGE_HANDLE && sourceKind !== "image") return "Only an Image node can feed the image input";
    if (sourceKind === "image" && this.kindOf(target) === "search" && targetHandle !== SEARCH_IMAGE_HANDLE) return "Wire the image into the Search node's image port (lower one)";
    return null;
  }
  /** source → [targets] adjacency over `edges` (shared by the graph walks). */
  adjacency(edges) {
    const adj = /* @__PURE__ */ new Map();
    for (const e of edges) {
      const list = adj.get(e.source);
      if (list) list.push(e.target);
      else adj.set(e.source, [e.target]);
    }
    return adj;
  }
  /** True if `from` can reach `to` by following `edges` (cycle guard). */
  reaches(edges, from, to) {
    const adj = this.adjacency(edges);
    const stack = [from];
    const seen = /* @__PURE__ */ new Set();
    while (stack.length) {
      const id = stack.pop();
      if (id === to) return true;
      if (seen.has(id)) continue;
      seen.add(id);
      for (const next of adj.get(id) ?? []) stack.push(next);
    }
    return false;
  }
  /** Node ids reachable DOWNSTREAM of `id` (its dependents) — used to confirm
   *  before deleting a node that feeds others. */
  dependentsOf(id) {
    const adj = this.adjacency(this.edges);
    const out = /* @__PURE__ */ new Set();
    const stack = [...adj.get(id) ?? []];
    while (stack.length) {
      const next = stack.pop();
      if (out.has(next)) continue;
      out.add(next);
      for (const n2 of adj.get(next) ?? []) stack.push(n2);
    }
    out.delete(id);
    return [...out];
  }
  /** The merged result set flowing INTO `id` from its direct predecessors' LAST
   *  run — the same union the executor feeds a node's scope. Reads each incoming
   *  edge's source `runtime.hits` and dedupes by chunk identity. Returns null
   *  when no predecessor has produced hits (so the Atlas modal shows all points).
   *  Used at EDIT time (open the Atlas modal pre-filtered to upstream results). */
  getPredecessorHits(id) {
    const merged = [];
    for (const e of this.edges) {
      if (e.target !== id) continue;
      const hits = this.runtime[e.source]?.hits;
      if (hits && hits.length) merged.push(...hits);
    }
    return merged.length ? dedupeHits(merged) : null;
  }
  // ── Undo / redo (auto-checkpointed via the canvas's debounced effect) ────────
  /** Called by the canvas after changes settle (debounced): push the PREVIOUS
   *  state so the whole settled change — structural OR config edits OR a move —
   *  becomes one undo step. Nothing is lost between checkpoints. */
  checkpoint(json) {
    if (this.lastCheckpoint === "") {
      this.lastCheckpoint = json;
      return;
    }
    if (json === this.lastCheckpoint) return;
    this.undoHistory.push(this.lastCheckpoint);
    this.lastCheckpoint = json;
  }
  undo() {
    if (this.running) return;
    const prev = this.undoHistory.undo(this.snapshot());
    if (prev === null) return;
    this.restore(prev);
    this.lastCheckpoint = prev;
  }
  redo() {
    if (this.running) return;
    const next = this.undoHistory.redo(this.snapshot());
    if (next === null) return;
    this.restore(next);
    this.lastCheckpoint = next;
  }
  /** Rebuild the graph from a snapshot string (undo/redo). Tolerant of a bad
   *  string (no-op). Preserves run results for surviving nodes so an unrelated
   *  structural undo doesn't wipe the displayed hits. */
  restore(json) {
    const parsed = safeParseGraph(json);
    if (!parsed) return;
    this.applyParsed(parsed, { preserveRuntime: true });
    this.selectedNodeIds = [];
    this.selectedEdgeIds = [];
    this.inspectedNodeId = null;
  }
  // ── Copy / paste + tidy + reconnect ─────────────────────────────────────────
  /** Copy the selected nodes (type + config + position) to the clipboard. */
  copySelection() {
    this.clipboard = this.selectedNodeIds.flatMap((id) => {
      const node = this.nodes.find((n2) => n2.id === id);
      const cfg = this.config[id];
      if (!node || !cfg || !isNodeKind(node.type)) return [];
      return [
        {
          type: node.type,
          config: { ...cfg, image: null },
          position: { ...node.position }
        }
      ];
    });
    this.pasteCount = 0;
  }
  /** Paste the clipboard nodes (fresh ids, cascading offset, selected on the
   *  canvas). Auto-checkpointed by the settle effect; blocked mid-run. */
  paste() {
    if (!this.clipboard.length || this.running) return;
    this.pasteCount += 1;
    const off = PASTE_OFFSET_PX * this.pasteCount;
    const config = { ...this.config };
    const runtime = { ...this.runtime };
    const newIds = /* @__PURE__ */ new Set();
    const newNodes = [];
    for (const item of this.clipboard) {
      const id = `${item.type}-${++this.seq}`;
      config[id] = { ...item.config, image: null };
      runtime[id] = blankRuntime();
      newNodes.push({
        id,
        type: item.type,
        position: { x: item.position.x + off, y: item.position.y + off },
        data: {},
        selected: true
      });
      newIds.add(id);
    }
    this.config = config;
    this.runtime = runtime;
    this.nodes = [
      ...this.nodes.map((n2) => ({ ...n2, selected: false })),
      ...newNodes
    ];
    this.selectedNodeIds = [...newIds];
  }
  /** Auto-layout the graph left-to-right. Blocked mid-run. elkjs is async, so
   *  the nodes update when layout resolves (fire-and-forget from the button). */
  async tidy() {
    if (this.running) return;
    this.nodes = await autoLayout(this.nodes, this.edges);
  }
  /** Patch one node's run state. */
  patchRuntime(id, patch) {
    const prev = this.runtime[id] ?? blankRuntime();
    this.runtime = { ...this.runtime, [id]: { ...prev, ...patch } };
  }
  /** Clear all run state back to idle (keeps the graph + user input). */
  resetRun() {
    const next = {};
    for (const n2 of this.nodes) next[n2.id] = blankRuntime();
    this.runtime = next;
    this.lastError = null;
  }
  /** Reset everything to the seeded starter graph. */
  reset() {
    this.seed();
  }
  /** Play a clicked result in the Inspector. */
  selectHit(hit) {
    this.selectedHit = hit;
  }
  /** Stop playing the selected result (back to the inspected node's results). */
  closeDetail() {
    this.selectedHit = null;
  }
  /** Show a node's intermediate state (config + results) in the Inspector.
   *  Clearing `selectedHit` lets a node click switch the panel away from a
   *  playing result. (Result-row clicks `stopPropagation`, so they never
   *  reach here and keep playing.) */
  inspectNode(id) {
    this.selectedHit = null;
    this.inspectedNodeId = id;
  }
  /** Record the canvas selection (from `<SvelteFlow onselectionchange>`). */
  setSelection(nodeIds, edgeIds) {
    this.selectedNodeIds = nodeIds;
    this.selectedEdgeIds = edgeIds;
  }
  /** Disconnect one edge (the nodes stay). */
  removeEdge(id) {
    if (this.running) return;
    this.edges = this.edges.filter((e) => e.id !== id);
    this.selectedEdgeIds = this.selectedEdgeIds.filter((x) => x !== id);
  }
  /** Delete one node, every edge touching it, and its config/runtime. */
  removeNode(id) {
    if (this.running) return;
    this.nodes = this.nodes.filter((n2) => n2.id !== id);
    this.edges = this.edges.filter((e) => e.source !== id && e.target !== id);
    const config = { ...this.config };
    const runtime = { ...this.runtime };
    delete config[id];
    delete runtime[id];
    this.config = config;
    this.runtime = runtime;
    this.selectedNodeIds = this.selectedNodeIds.filter((x) => x !== id);
    const liveEdges = new Set(this.edges.map((e) => e.id));
    this.selectedEdgeIds = this.selectedEdgeIds.filter((x) => liveEdges.has(x));
    if (this.inspectedNodeId === id) this.inspectedNodeId = null;
  }
  /** Delete the current selection: selected nodes (and their edges) + edges. */
  deleteSelected() {
    const nodeIds = new Set(this.selectedNodeIds);
    const edgeIds = new Set(this.selectedEdgeIds);
    if (!nodeIds.size && !edgeIds.size || this.running) return;
    this.nodes = this.nodes.filter((n2) => !nodeIds.has(n2.id));
    this.edges = this.edges.filter((e) => !edgeIds.has(e.id) && !nodeIds.has(e.source) && !nodeIds.has(e.target));
    const config = { ...this.config };
    const runtime = { ...this.runtime };
    for (const id of nodeIds) {
      delete config[id];
      delete runtime[id];
    }
    this.config = config;
    this.runtime = runtime;
    if (this.inspectedNodeId && nodeIds.has(this.inspectedNodeId)) this.inspectedNodeId = null;
    this.selectedNodeIds = [];
    this.selectedEdgeIds = [];
  }
  /** Prune state for elements Svelte Flow already removed itself — its
   *  built-in Backspace/Delete mutates the bound `nodes`/`edges` directly, so
   *  it never reaches `removeNode`/`deleteSelected`. Wired via `<SvelteFlow
   *  ondelete>` so all three delete paths converge on the same cleanup. */
  syncDeleted(nodeIds, edgeIds) {
    if (!nodeIds.length && !edgeIds.length) return;
    const gone = new Set(nodeIds);
    const config = { ...this.config };
    const runtime = { ...this.runtime };
    for (const id of gone) {
      delete config[id];
      delete runtime[id];
    }
    this.config = config;
    this.runtime = runtime;
    if (this.inspectedNodeId && gone.has(this.inspectedNodeId)) this.inspectedNodeId = null;
    this.selectedNodeIds = this.selectedNodeIds.filter((x) => !gone.has(x));
    const goneEdges = new Set(edgeIds);
    this.selectedEdgeIds = this.selectedEdgeIds.filter((x) => !goneEdges.has(x));
  }
  // ── Persistence ───────────────────────────────────────────────────────────
  /** JSON snapshot of the serialisable graph. Reads nodes/edges/config deeply,
   *  so calling it inside a `$effect` tracks every change (drives autosave). */
  snapshot() {
    const nodes = this.nodes.flatMap((n2) => isNodeKind(n2.type) ? [
      {
        id: n2.id,
        type: n2.type,
        position: { x: n2.position.x, y: n2.position.y }
      }
    ] : []);
    const edges = this.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      ...typeof e.targetHandle === "string" ? { targetHandle: e.targetHandle } : {},
      ...typeof e.label === "string" ? { label: e.label } : {}
    }));
    const config = {};
    for (const [id, c] of Object.entries(this.config)) {
      config[id] = {
        q: c.q,
        imageName: c.imageName,
        where: c.where,
        filters: c.filters,
        // A generic (any-key) mode narrows to the persisted set here; an unknown
        // key self-heals to 'fts' on reload (persistence.ts picklist fallback).
        mode: c.mode,
        n: c.n,
        rerank: c.rerank,
        minScore: c.minScore,
        refineScope: c.refineScope,
        combineMode: c.combineMode,
        tags: c.tags,
        exportFormat: c.exportFormat,
        exportColumns: c.exportColumns,
        // Never round-trip the captured Atlas selection (heavy Hit[]); see
        // persistence.ts — a reload discards it (re-open the modal to re-select).
        capturedAtlasSelection: null,
        label: c.label,
        enabled: c.enabled
      };
    }
    return JSON.stringify({ nodes, edges, config, tags: this.tags.snapshot() });
  }
  /** Write a snapshot string to localStorage (no-op outside the browser). */
  persist(json) {
    return;
  }
  /** Rehydrate from localStorage; returns false (→ caller seeds) if absent/bad.
   *  A bad node shape / unknown kind fails the parse, so we seed instead of
   *  crashing the canvas. */
  load() {
    return false;
  }
  /** Rebuild nodes/edges/config/runtime/tags from a parsed graph — shared by
   *  load() (localStorage) and restore() (undo/redo). `preserveRuntime` keeps
   *  existing run results for surviving node ids (undo/redo shouldn't wipe hits). */
  applyParsed(parsed, opts = {}) {
    const nodes = parsed.nodes.map((n2) => ({
      id: n2.id,
      type: n2.type,
      position: { x: n2.position.x, y: n2.position.y },
      data: {}
    }));
    const ids = new Set(nodes.map((n2) => n2.id));
    const kindById = new Map(parsed.nodes.map((n2) => [n2.id, n2.type]));
    const searchPort = (e) => e.targetHandle ?? (kindById.get(e.source) === "image" ? SEARCH_IMAGE_HANDLE : SEARCH_IN_HANDLE);
    const edges = parsed.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      ...kindById.get(e.target) === "search" ? { targetHandle: searchPort(e) } : {},
      ...e.label ? { label: e.label } : {}
    }));
    const config = {};
    const runtime = {};
    for (const n2 of nodes) {
      const pc = parsed.config[n2.id];
      config[n2.id] = pc ? { ...pc, image: null } : defaultConfig();
      runtime[n2.id] = opts.preserveRuntime ? this.runtime[n2.id] ?? blankRuntime() : blankRuntime();
    }
    this.nodes = nodes;
    this.edges = edges;
    this.config = config;
    this.runtime = runtime;
    this.tags.hydrate(parsed.tags);
    this.seq = this.maxSeq();
  }
  /** Highest numeric suffix across node ids (so new ids never collide). */
  maxSeq() {
    let max = 0;
    for (const n2 of this.nodes) {
      const m = /-(\d+)$/.exec(n2.id);
      if (m) max = Math.max(max, Number(m[1]));
    }
    return max;
  }
  // ── Execution (delegated to executor.ts) ────────────────────────────────
  /** The executor's narrow view of our state + the runtime writers it needs.
   *  The executor owns the dataflow algorithm; the store owns the state. */
  runDeps() {
    return {
      nodes: this.nodes,
      edges: this.edges,
      config: (id) => this.config[id] ?? defaultConfig(),
      kindOf: (id) => this.kindOf(id),
      patchRuntime: (id, patch) => this.patchRuntime(id, patch),
      tagHits: (hits, tags) => this.tags.addTo(hits, tags),
      cachedOutput: (id) => {
        const rt = this.runtime[id];
        const out = rt?.output;
        if (!out || out.failed || rt.stale) return null;
        if (rt.outputKey !== this.nodeFingerprint(id)) return null;
        return out;
      },
      fingerprint: (id) => this.nodeFingerprint(id)
    };
  }
  /** Run the whole graph from scratch (the toolbar Run button). */
  async run() {
    if (this.running) return;
    this.running = true;
    this.resetRun();
    try {
      const error = await runGraph(this.runDeps());
      if (error) this.lastError = error;
    } finally {
      this.running = false;
    }
  }
  /** Run ONE node (the node ▶ button): recomputes the node itself, reuses
   *  upstream results where they exist, and computes missing/stale/failed
   *  upstream once. `fresh` re-executes the whole upstream branch. Everything
   *  else keeps its results but is flagged stale when it now sits downstream
   *  of fresher data. */
  async runNode(id, opts = {}) {
    if (this.running) return;
    this.running = true;
    this.lastError = null;
    try {
      const { error, ran } = await runSubgraph(this.runDeps(), id, opts);
      if (error) {
        this.lastError = error;
        return;
      }
      const ranSet = new Set(ran);
      const staleIds = /* @__PURE__ */ new Set();
      for (const r of ran) {
        for (const d of this.dependentsOf(r)) if (!ranSet.has(d)) staleIds.add(d);
      }
      for (const d of staleIds) {
        const rt = this.runtime[d];
        if (rt && (rt.status !== "idle" || rt.output)) this.patchRuntime(d, { stale: true });
      }
    } finally {
      this.running = false;
    }
  }
}
const graph = new WorkflowGraph();
const FIELD_CLASS = "nodrag rounded border border-border bg-background px-2 py-1 text-xs text-foreground outline-none focus:border-primary";
function NodeShell($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      id,
      title,
      status = "idle",
      selected = false,
      width = "w-64",
      children
    } = $$props;
    const cfg = derived(() => graph.config[id]);
    const displayTitle = derived(() => cfg()?.label?.trim() || title);
    const enabled = derived(() => cfg()?.enabled ?? true);
    const stale = derived(() => (graph.runtime[id]?.stale ?? false) || graph.isOutdated(id));
    const error = derived(() => status === "error" ? graph.runtime[id]?.error ?? null : null);
    const btn = "nodrag shrink-0 rounded p-0.5 text-muted-foreground/40 opacity-0 transition-opacity group-hover:opacity-100 hover:text-foreground";
    $$renderer2.push(`<div${attr_class(`group ${stringify(width)} rounded-lg border border-border bg-card shadow-sm transition-all`, void 0, {
      "ring-2": selected,
      "ring-primary": selected,
      "opacity-60": !enabled()
    })}><div class="flex items-center gap-1.5 border-b border-border px-3 py-1.5"><span${attr_class(`size-2 shrink-0 rounded-full ${stringify(STATUS_DOT[status])}`)}></span> <span class="min-w-0 flex-1 truncate text-xs font-semibold text-foreground"${attr("title", displayTitle())}>${escape_html(displayTitle())}</span> `);
    if (!enabled()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<span class="shrink-0 rounded bg-muted px-1 text-[9px] tracking-wide text-muted-foreground uppercase">off</span>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (stale()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<span class="shrink-0 rounded bg-amber-500/15 px-1 text-[9px] tracking-wide text-amber-600 uppercase dark:text-amber-400" title="This node's results are out of date (it was edited, rewired, or upstream re-ran) — press ▶ to refresh">stale</span>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> <button${attr_class(`${stringify(btn)} disabled:opacity-30`)} title="Run this node — reuses upstream results, runs missing upstream once (Shift: rerun the whole branch)" aria-label="Run node"${attr("disabled", graph.running, true)}>`);
    Play($$renderer2, { class: "size-3.5" });
    $$renderer2.push(`<!----></button> <button${attr_class(clsx(btn))} title="Duplicate node" aria-label="Duplicate node">`);
    Copy($$renderer2, { class: "size-3.5" });
    $$renderer2.push(`<!----></button> <button${attr_class(`${stringify(btn)} disabled:opacity-30`)}${attr("title", enabled() ? "Disable (bypass) node" : "Enable node")} aria-label="Toggle node enabled"${attr("disabled", graph.running, true)}>`);
    if (enabled()) {
      $$renderer2.push("<!--[0-->");
      Eye_off($$renderer2, { class: "size-3.5" });
    } else {
      $$renderer2.push("<!--[-1-->");
      Eye($$renderer2, { class: "size-3.5" });
    }
    $$renderer2.push(`<!--]--></button> <button${attr_class(`${stringify(btn)} hover:bg-destructive/15 hover:text-destructive`)} title="Delete node" aria-label="Delete node">`);
    X($$renderer2, { class: "size-3.5" });
    $$renderer2.push(`<!----></button></div> <div class="px-3 py-2 text-xs text-foreground">`);
    if (error()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="nodrag mb-2 max-h-16 overflow-y-auto rounded border border-destructive/30 bg-destructive/10 px-2 py-1 text-[10px] leading-snug break-words text-destructive"${attr("title", error())}>${escape_html(error())}</div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> `);
    children($$renderer2);
    $$renderer2.push(`<!----></div></div>`);
  });
}
function QueryNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const cfg = derived(() => graph.config[id]);
    const rt = derived(() => graph.runtime[id]);
    if (cfg() && rt()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Text query",
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          $$renderer3.push(`<label class="mb-1 block text-[10px] text-muted-foreground"${attr("for", `q-${stringify(id)}`)}>Query text</label> <input${attr("id", `q-${stringify(id)}`)}${attr_class(`${stringify(FIELD_CLASS)} w-full`)} placeholder="e.g. Sverige"${attr("value", cfg().q)}/> <p class="mt-1 text-[10px] text-muted-foreground">Drives FTS / vector legs of Search.</p> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function ImageNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const cfg = derived(() => graph.config[id]);
    const rt = derived(() => graph.runtime[id]);
    if (
      // Revoke the live preview URL when this node unmounts (delete / reset), so
      // blob URLs don't leak across add → preview → delete cycles.
      cfg() && rt()
    ) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Image",
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          $$renderer3.push(`<input type="file" accept="image/*" class="nodrag w-full text-[10px] text-muted-foreground file:mr-2 file:rounded file:border-0 file:bg-secondary file:px-2 file:py-1 file:text-[10px] file:text-secondary-foreground"/> `);
          if (cfg().imageName) {
            $$renderer3.push("<!--[1-->");
            $$renderer3.push(`<p class="mt-2 text-[10px] text-amber-500">Previously: ${escape_html(cfg().imageName)} — re-upload (images aren't saved across reloads).</p>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> <p class="mt-1 text-[10px] text-muted-foreground">Wire into a Search's <span class="text-violet-400">img</span> port, then set mode = <span class="text-foreground">Image</span> or <span class="text-foreground">All</span>.</p> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function FilterNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const cfg = derived(() => graph.config[id]);
    const rt = derived(() => graph.runtime[id]);
    const view = derived(activeView);
    const fields = derived(() => view().filterFields.map((field) => ({
      field,
      label: view().metadataFields.find((m) => m.field === field)?.label ?? field
    })));
    if (cfg() && rt()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Filter",
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          $$renderer3.push(`<div class="flex flex-col gap-2"><!--[-->`);
          const each_array = ensure_array_like(fields());
          for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
            let f = each_array[$$index];
            $$renderer3.push(`<div><label class="mb-1 block text-[10px] text-muted-foreground"${attr("for", `${stringify(f.field)}-${stringify(id)}`)}>${escape_html(f.label)}</label> <input${attr("id", `${stringify(f.field)}-${stringify(id)}`)}${attr_class(`${stringify(FIELD_CLASS)} w-full`)}${attr("placeholder", `${stringify(f.label.toLowerCase())} contains…`)}${attr("value", cfg().filters[f.field] ?? "")}/></div>`);
          }
          $$renderer3.push(`<!--]--> <div><label class="mb-1 block text-[10px] text-muted-foreground"${attr("for", `where-${stringify(id)}`)}>SQL where</label> <input${attr("id", `where-${stringify(id)}`)}${attr_class(`${stringify(FIELD_CLASS)} w-full font-mono`)} placeholder="duration > 30"${attr("value", cfg().where)}/></div></div> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function AtlasNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const cfg = derived(() => graph.config[id]);
    const rt = derived(() => graph.runtime[id]);
    const capturedCount = derived(() => cfg()?.capturedAtlasSelection?.length ?? 0);
    if (cfg() && rt()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Atlas selection",
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          $$renderer3.push(`<div class="flex flex-col gap-2"><p class="text-sm font-medium text-foreground">${escape_html(capturedCount().toLocaleString())} points selected</p> <button type="button" class="nodrag inline-flex items-center justify-center gap-1.5 rounded border border-border bg-background px-2 py-1 text-[11px] font-medium text-foreground transition-colors hover:bg-muted">`);
          Map$1($$renderer3, { class: "size-3" });
          $$renderer3.push(`<!----> Open atlas</button> `);
          if (capturedCount() === 0) {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<p class="text-[10px] text-muted-foreground">Open the <span class="text-amber-500">Atlas</span> viewer, lasso a region, and <span class="text-foreground">Confirm</span> to capture it into the workflow.</p>`);
          } else {
            $$renderer3.push("<!--[-1-->");
            $$renderer3.push(`<p class="text-[10px] text-emerald-500">Selection captured. Wire into Search to refine within it.</p>`);
          }
          $$renderer3.push(`<!--]--></div> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> `);
    {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function SearchNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const cfg = derived(() => graph.config[id]);
    const rt = derived(() => graph.runtime[id]);
    const isVisual = derived(() => cfg()?.mode === "visual");
    const REFINE_SCOPES = [
      { value: "video", label: "Videos" },
      { value: "chunk", label: "Chunks" }
    ];
    const RESULT_KINDS = ["search", "combine", "tagger"];
    const hasUpstreamResults = derived(() => graph.edges.some((e) => {
      const k2 = graph.kindOf(e.source);
      return e.target === id && k2 !== null && RESULT_KINDS.includes(k2);
    }));
    if (cfg() && rt()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: `Search · ${stringify(modeLabel(cfg().mode))}`,
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          Handle($$renderer3, {
            id: SEARCH_IN_HANDLE,
            type: "target",
            position: Position.Left,
            style: "top: 35%",
            title: "Query · filter · upstream results"
          });
          $$renderer3.push(`<!----> `);
          Handle($$renderer3, {
            id: SEARCH_IMAGE_HANDLE,
            type: "target",
            position: Position.Left,
            style: "top: 65%; background: #8b5cf6; border-color: #8b5cf6;",
            title: "Image (wire an Image node here)"
          });
          $$renderer3.push(`<!----> <span class="pointer-events-none absolute left-1.5 -translate-y-1/2 text-[8px] leading-none text-muted-foreground" style="top: 35%">in</span> <span class="pointer-events-none absolute left-1.5 -translate-y-1/2 text-[8px] leading-none text-violet-400" style="top: 65%">img</span> <label class="mb-1 block text-[10px] text-muted-foreground"${attr("for", `q-${stringify(id)}`)}>${escape_html(isVisual() ? "Query (optional — image drives it)" : "Query")}</label> <input${attr("id", `q-${stringify(id)}`)}${attr_class(`${stringify(FIELD_CLASS)} w-full`)}${attr("placeholder", isVisual() ? "uses the connected image" : "e.g. skatt")}${attr("value", cfg().q)}/> <label class="mt-2 mb-1 block text-[10px] text-muted-foreground"${attr("for", `mode-${stringify(id)}`)}>Mode</label> `);
          $$renderer3.select(
            {
              id: `mode-${stringify(id)}`,
              class: `${stringify(FIELD_CLASS)} w-full`,
              value: cfg().mode,
              onchange: (e) => graph.setConfig(id, { mode: e.currentTarget.value })
            },
            ($$renderer4) => {
              $$renderer4.push(`<!--[-->`);
              const each_array = ensure_array_like(SEARCH_MODES);
              for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                let m = each_array[$$index];
                $$renderer4.option({ value: m.value }, ($$renderer5) => {
                  $$renderer5.push(`${escape_html(m.label)}`);
                });
              }
              $$renderer4.push(`<!--]-->`);
            }
          );
          $$renderer3.push(` <div class="mt-2 flex items-center gap-2"><label class="text-[10px] text-muted-foreground"${attr("for", `n-${stringify(id)}`)}>Results</label> <input${attr("id", `n-${stringify(id)}`)} type="number"${attr("min", MIN_N)}${attr("max", MAX_N)}${attr_class(`${stringify(FIELD_CLASS)} w-16`)}${attr("value", cfg().n)}/> <label class="nodrag ml-auto flex items-center gap-1.5 text-[10px] text-muted-foreground"><input type="checkbox"${attr("checked", cfg().rerank, true)}/> Rerank</label></div> <div class="mt-2 flex items-center gap-2"><label class="text-[10px] text-muted-foreground"${attr("for", `min-score-${stringify(id)}`)} title="Drop hits scoring below this (normalized, higher = better). Empty = no threshold.">Min score</label> <input${attr("id", `min-score-${stringify(id)}`)} type="number" step="any"${attr_class(`${stringify(FIELD_CLASS)} w-16`)} placeholder="off"${attr("value", cfg().minScore ?? "")}/></div> `);
          if (hasUpstreamResults()) {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<div class="nodrag mt-2 flex items-center gap-1.5 text-[10px] text-muted-foreground"><span title="Re-rank all chunks in the upstream videos, or narrow to the exact upstream chunks">Refine within</span> <div class="ml-auto flex overflow-hidden rounded border border-border"><!--[-->`);
            const each_array_1 = ensure_array_like(REFINE_SCOPES);
            for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
              let s = each_array_1[$$index_1];
              $$renderer3.push(`<button type="button"${attr_class(`px-1.5 py-0.5 transition-colors ${stringify(cfg().refineScope === s.value ? "bg-primary text-primary-foreground" : "hover:bg-muted")}`)}>${escape_html(s.label)}</button>`);
            }
            $$renderer3.push(`<!--]--></div></div>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          if (rt().status !== "error") {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<div class="mt-2 border-t border-border pt-1.5 text-[10px]">`);
            if (rt().status === "running") {
              $$renderer3.push("<!--[0-->");
              $$renderer3.push(`<span class="text-primary">Searching…</span>`);
            } else if (rt().status === "done") {
              $$renderer3.push("<!--[1-->");
              $$renderer3.push(`<span class="text-muted-foreground"><span class="text-foreground">${escape_html(rt().count)}</span> hits · ${escape_html(rt().ms)} ms`);
              if (rt().scopedDocs) {
                $$renderer3.push("<!--[0-->");
                $$renderer3.push(`· within <span class="text-foreground">${escape_html(rt().scopedDocs)}</span> videos`);
                if (rt().scopeCapped) {
                  $$renderer3.push("<!--[0-->");
                  $$renderer3.push(`<span class="text-amber-500">(capped)</span>`);
                } else {
                  $$renderer3.push("<!--[-1-->");
                }
                $$renderer3.push(`<!--]-->`);
              } else if (rt().scopedChunks) {
                $$renderer3.push("<!--[1-->");
                $$renderer3.push(`· within <span class="text-foreground">${escape_html(rt().scopedChunks)}</span> chunks`);
                if (rt().scopeCapped) {
                  $$renderer3.push("<!--[0-->");
                  $$renderer3.push(`<span class="text-amber-500">(capped)</span>`);
                } else {
                  $$renderer3.push("<!--[-1-->");
                }
                $$renderer3.push(`<!--]-->`);
              } else {
                $$renderer3.push("<!--[-1-->");
              }
              $$renderer3.push(`<!--]--></span> `);
              if (rt().count === 0 && (rt().scopedDocs || rt().scopedChunks)) {
                $$renderer3.push("<!--[0-->");
                $$renderer3.push(`<div class="mt-1 text-amber-500">Nothing matched inside the ${escape_html(rt().scopedChunks ? `${rt().scopedChunks} chunks` : `${rt().scopedDocs} videos`)}
              — delete the incoming refine edge to search all.</div>`);
              } else {
                $$renderer3.push("<!--[-1-->");
              }
              $$renderer3.push(`<!--]-->`);
            } else {
              $$renderer3.push("<!--[-1-->");
              $$renderer3.push(`<span class="text-muted-foreground/70">idle — add a query or image, then Run</span>`);
            }
            $$renderer3.push(`<!--]--> `);
            if (rt().droppedInputs > 0) {
              $$renderer3.push("<!--[0-->");
              $$renderer3.push(`<div class="mt-1 text-amber-500">${escape_html(rt().droppedInputs)} extra input${escape_html(rt().droppedInputs > 1 ? "s" : "")} ignored — a Search uses one
            query + one image. Use a Combine node to merge result sets.</div>`);
            } else {
              $$renderer3.push("<!--[-1-->");
            }
            $$renderer3.push(`<!--]--></div>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function CombineNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const cfg = derived(() => graph.config[id]);
    const rt = derived(() => graph.runtime[id]);
    const MODES = [
      { value: "union", label: "∪ Union" },
      { value: "intersect", label: "∩ Intersect" }
    ];
    if (cfg() && rt()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Combine",
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          Handle($$renderer3, { type: "target", position: Position.Left });
          $$renderer3.push(`<!----> <div class="flex gap-1"><!--[-->`);
          const each_array = ensure_array_like(MODES);
          for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
            let m = each_array[$$index];
            $$renderer3.push(`<button type="button"${attr_class(`nodrag flex-1 rounded border px-2 py-1 text-[11px] transition-colors ${stringify(cfg().combineMode === m.value ? "border-primary bg-primary/10 text-foreground" : "border-border text-muted-foreground hover:bg-muted")}`)}>${escape_html(m.label)}</button>`);
          }
          $$renderer3.push(`<!--]--></div> <p class="mt-1 text-[10px] text-muted-foreground">${escape_html(cfg().combineMode === "intersect" ? "Keep only chunks present in ALL inputs." : "Keep chunks present in ANY input.")}</p> `);
          if (rt().status !== "error") {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<div class="mt-2 border-t border-border pt-1.5 text-[10px]">`);
            if (rt().status === "done") {
              $$renderer3.push("<!--[0-->");
              $$renderer3.push(`<span class="text-muted-foreground"><span class="text-foreground">${escape_html(rt().count)}</span> combined</span>`);
            } else {
              $$renderer3.push("<!--[-1-->");
              $$renderer3.push(`<span class="text-muted-foreground/70">idle — wire 2+ result sets in, then Run</span>`);
            }
            $$renderer3.push(`<!--]--></div>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function TaggerNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const cfg = derived(() => graph.config[id]);
    const rt = derived(() => graph.runtime[id]);
    let draft = "";
    if (cfg() && rt()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Tagger",
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          Handle($$renderer3, { type: "target", position: Position.Left });
          $$renderer3.push(`<!----> <label class="mb-1 block text-[10px] text-muted-foreground"${attr("for", `tag-${stringify(id)}`)}>Tags stamped on every hit</label> <div class="mb-1.5 flex flex-wrap gap-1"><!--[-->`);
          const each_array = ensure_array_like(cfg().tags);
          for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
            let tag = each_array[$$index];
            $$renderer3.push(`<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-foreground">${escape_html(tag)} <button type="button" class="nodrag text-muted-foreground hover:text-destructive"${attr("aria-label", `Remove tag ${stringify(tag)}`)}>`);
            X($$renderer3, { class: "size-3" });
            $$renderer3.push(`<!----></button></span>`);
          }
          $$renderer3.push(`<!--]--> `);
          if (cfg().tags.length === 0) {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<span class="text-[10px] text-muted-foreground/70">no tags yet</span>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--></div> <input${attr("id", `tag-${stringify(id)}`)}${attr_class(`${stringify(FIELD_CLASS)} nodrag w-full`)} placeholder="type a tag, press Enter"${attr("value", draft)}/> `);
          if (rt().status !== "error") {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<div class="mt-2 border-t border-border pt-1.5 text-[10px]">`);
            if (rt().status === "done") {
              $$renderer3.push("<!--[0-->");
              $$renderer3.push(`<span class="text-muted-foreground">tagged <span class="text-foreground">${escape_html(rt().count)}</span> hits</span>`);
            } else {
              $$renderer3.push("<!--[-1-->");
              $$renderer3.push(`<span class="text-muted-foreground/70">idle — wire results in, then Run</span>`);
            }
            $$renderer3.push(`<!--]--></div>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function HitList($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { hits, maxHeight = "max-h-72" } = $$props;
    const selectedKey = derived(() => graph.selectedHit ? hitKey(graph.selectedHit) : null);
    let addingKey = null;
    let draft = "";
    $$renderer2.push(`<div${attr_class(`nodrag nowheel flex ${stringify(maxHeight)} flex-col gap-1.5 overflow-y-auto pr-1`)}><!--[-->`);
    const each_array = ensure_array_like(hits);
    for (let $$index_1 = 0, $$length = each_array.length; $$index_1 < $$length; $$index_1++) {
      let h2 = each_array[$$index_1];
      const key2 = hitKey(h2);
      const isSel = selectedKey() === key2;
      const tags = graph.tags.forHit(h2);
      const title = activeView().title(h2);
      const body = activeView().body(h2);
      $$renderer2.push(`<div${attr_class("group rounded border bg-background transition-colors", void 0, { "border-primary": isSel, "border-border": !isSel })}><button type="button" class="flex w-full gap-2 p-1.5 text-left transition-colors hover:bg-muted"><div class="relative shrink-0"><img${attr("src", chunkFrameUrl(h2))} alt="" loading="lazy" class="h-10 w-14 rounded bg-muted object-cover" onerror="this.__e=event"/> <span class="pointer-events-none absolute inset-0 grid place-items-center rounded bg-black/45 opacity-0 transition-opacity group-hover:opacity-100">`);
      Play($$renderer2, { class: "size-4 text-white" });
      $$renderer2.push(`<!----></span></div> <div class="min-w-0">`);
      if (title) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="truncate text-[10px] font-medium text-foreground"${attr("title", title)}>${escape_html(title)}</div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> <div class="line-clamp-2 text-[10px] text-muted-foreground">${escape_html(body)}</div></div></button> <div class="flex flex-wrap items-center gap-1 px-1.5 pb-1.5"><!--[-->`);
      const each_array_1 = ensure_array_like(tags);
      for (let $$index = 0, $$length2 = each_array_1.length; $$index < $$length2; $$index++) {
        let tag = each_array_1[$$index];
        $$renderer2.push(`<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-1.5 py-0.5 text-[9px] text-foreground">${escape_html(tag)} <button type="button" class="text-muted-foreground hover:text-destructive"${attr("aria-label", `Remove tag ${stringify(tag)}`)}>`);
        X($$renderer2, { class: "size-2.5" });
        $$renderer2.push(`<!----></button></span>`);
      }
      $$renderer2.push(`<!--]--> `);
      if (addingKey === key2) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<input class="w-24 rounded border border-border bg-background px-1 py-0.5 text-[9px] text-foreground outline-none focus:border-primary" placeholder="tag…" autofocus=""${attr("value", draft)}/>`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<button type="button" class="inline-flex items-center gap-0.5 rounded border border-dashed border-border px-1 py-0.5 text-[9px] text-muted-foreground hover:bg-muted hover:text-foreground">`);
        Plus$1($$renderer2, { class: "size-2.5" });
        $$renderer2.push(`<!----> tag</button>`);
      }
      $$renderer2.push(`<!--]--></div></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function ResultsNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const rt = derived(() => graph.runtime[id]);
    const hits = derived(() => rt()?.hits ?? []);
    if (rt()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Results",
        status: rt().status,
        selected,
        width: "w-80",
        children: ($$renderer3) => {
          Handle($$renderer3, { type: "target", position: Position.Left });
          $$renderer3.push(`<!----> `);
          if (rt().status === "idle") {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<p class="text-[11px] text-muted-foreground">Run the graph to see hits here.</p>`);
          } else if (hits().length === 0) {
            $$renderer3.push("<!--[1-->");
            $$renderer3.push(`<p class="text-[11px] text-muted-foreground">No results.</p>`);
          } else {
            $$renderer3.push("<!--[-1-->");
            $$renderer3.push(`<div class="mb-1.5 text-[10px] text-muted-foreground"><span class="text-foreground">${escape_html(hits().length)}</span> results · click to play</div> `);
            HitList($$renderer3, { hits: hits() });
            $$renderer3.push(`<!---->`);
          }
          $$renderer3.push(`<!--]--> `);
          Handle($$renderer3, { type: "source", position: Position.Right });
          $$renderer3.push(`<!---->`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function exportColumns() {
  const view = activeView();
  const { declared } = view.descriptor;
  const cols = [...view.keyFields];
  if (declared.time) cols.push(declared.time.start, declared.time.end, "duration");
  if (view.bodyField) cols.push(view.bodyField);
  if (view.captionField) cols.push(view.captionField);
  for (const { field } of view.metadataFields) cols.push(field);
  cols.push("_score", "tags");
  return [...new Set(cols)];
}
function ExportNode($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { id, selected } = $$props;
    const rt = derived(() => graph.runtime[id]);
    const cfg = derived(() => graph.config[id]);
    const hits = derived(() => rt()?.hits ?? []);
    const cols = derived(() => cfg() ? cfg().exportColumns ?? exportColumns() : []);
    const canDownload = derived(() => hits().length > 0 && cols().length > 0);
    if (rt() && cfg()) {
      $$renderer2.push("<!--[0-->");
      NodeShell($$renderer2, {
        id,
        title: "Export",
        status: rt().status,
        selected,
        children: ($$renderer3) => {
          Handle($$renderer3, { type: "target", position: Position.Left });
          $$renderer3.push(`<!----> <div class="flex flex-col gap-2"><div class="text-[11px] text-muted-foreground">`);
          if (hits().length) {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<span class="text-foreground">${escape_html(hits().length)}</span> hits ·
          ${escape_html(cols().length)} col${escape_html(cols().length === 1 ? "" : "s")} ·
          ${escape_html(cfg().exportFormat.toUpperCase())} `);
            if (cols().includes("tags")) {
              $$renderer3.push("<!--[0-->");
              $$renderer3.push(`<span class="text-primary">+ tags</span>`);
            } else {
              $$renderer3.push("<!--[-1-->");
            }
            $$renderer3.push(`<!--]-->`);
          } else if (rt().status === "idle") {
            $$renderer3.push("<!--[1-->");
            $$renderer3.push(`Run the graph to feed this export.`);
          } else {
            $$renderer3.push("<!--[-1-->");
            $$renderer3.push(`No results — check upstream nodes.`);
          }
          $$renderer3.push(`<!--]--></div> <button type="button" class="nodrag inline-flex items-center justify-center gap-1.5 rounded border border-border bg-background px-2 py-1 text-[11px] font-medium text-foreground transition-colors hover:bg-muted disabled:opacity-50"${attr("disabled", !canDownload(), true)}>`);
          Download($$renderer3, { class: "size-3" });
          $$renderer3.push(`<!----> Download ${escape_html(cfg().exportFormat.toUpperCase())}</button></div>`);
        }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
const nodeTypes = {
  query: QueryNode,
  image: ImageNode,
  filter: FilterNode,
  atlas: AtlasNode,
  search: SearchNode,
  combine: CombineNode,
  tagger: TaggerNode,
  results: ResultsNode,
  export: ExportNode
};
const ARROW_COLOR = "#94a3b8";
const ARROW_SIZE = 18;
const ARROW_MARKER = {
  type: MarkerType.ArrowClosed,
  width: ARROW_SIZE,
  height: ARROW_SIZE,
  color: ARROW_COLOR
};
const PAYLOAD_COLOR = {
  query: "#f59e0b",
  // amber  — text query spec
  image: "#8b5cf6",
  // violet — image spec
  filter: "#64748b",
  // slate  — metadata filter spec
  atlas: "#10b981",
  // emerald — atlas selection result set
  results: "#10b981",
  // emerald — a concrete result set
  refine: "#10b981",
  // emerald — results used to scope a downstream Search
  tagged: "#22c55e",
  // green   — tagged results
  default: ARROW_COLOR
};
function edgePayload(source, target) {
  let label;
  if (source === "query" || source === "image" || source === "filter") label = source;
  else if (source === "atlas") label = target === "search" ? "refine" : "results";
  else if (source === "tagger") label = "tagged";
  else if (source === "search" || source === "combine")
    label = target === "search" ? "refine" : "results";
  else label = "results";
  return { label, color: PAYLOAD_COLOR[label] ?? PAYLOAD_COLOR.default };
}
function ReconnectableEdge($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      id,
      source,
      target,
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      markerEnd,
      label,
      selected
    } = $$props;
    let $$d = derived(() => getBezierPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition
    })), $$derived_array = derived(() => to_array($$d(), 3)), path = derived(() => $$derived_array()[0]), labelX = derived(() => $$derived_array()[1]), labelY = derived(() => $$derived_array()[2]);
    const payload = derived(() => edgePayload(graph.kindOf(source), graph.kindOf(target)));
    const running = derived(() => graph.runtime[target]?.status === "running");
    const edgeLabel = derived(() => label ?? payload().label);
    const style = derived(() => `stroke: ${payload().color}; stroke-width: 1.5;` + (running() ? " stroke-dasharray: 6; animation: edge-dash 0.7s linear infinite;" : ""));
    BaseEdge($$renderer2, spread_props([
      {
        path: path(),
        labelX: labelX(),
        labelY: labelY(),
        label: edgeLabel(),
        style: style()
      },
      markerEnd !== void 0 ? { markerEnd } : {}
    ]));
    $$renderer2.push(`<!----> `);
    EdgeReconnectAnchor($$renderer2, { type: "source", position: { x: sourceX, y: sourceY } });
    $$renderer2.push(`<!----> `);
    EdgeReconnectAnchor($$renderer2, { type: "target", position: { x: targetX, y: targetY } });
    $$renderer2.push(`<!----> `);
    if (selected) {
      $$renderer2.push("<!--[0-->");
      EdgeLabel($$renderer2, {
        x: labelX(),
        y: labelY() - 18,
        children: ($$renderer3) => {
          $$renderer3.push(`<button type="button" class="grid size-5 place-items-center rounded-full border border-border bg-card text-muted-foreground shadow transition-colors hover:bg-destructive/15 hover:text-destructive" title="Disconnect edge" aria-label="Disconnect edge">`);
          X($$renderer3, { class: "size-3" });
          $$renderer3.push(`<!----></button>`);
        },
        $$slots: { default: true }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
class CommandMenuState {
  open = false;
  toggle() {
    this.open = !this.open;
  }
}
const commandMenu = new CommandMenuState();
function WorkflowToolbar($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<div class="flex flex-col gap-2 rounded-lg border border-border bg-card/95 p-2 shadow-md backdrop-blur"><div class="flex items-center gap-1.5">`);
    Button($$renderer2, {
      size: "sm",
      onclick: (
        /** Canvas run controls (Run / Clear / Reset / Delete / Undo / Redo / Tidy).
        *  Adding nodes lives in the drag-to-add palette, not here. */
        () => graph.run()
      ),
      disabled: graph.running,
      children: ($$renderer3) => {
        if (graph.running) {
          $$renderer3.push("<!--[0-->");
          Loader_circle($$renderer3, { class: "size-3.5 animate-spin" });
        } else {
          $$renderer3.push("<!--[-1-->");
          Play($$renderer3, { class: "size-3.5" });
        }
        $$renderer3.push(`<!--]--> Run`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> `);
    Button($$renderer2, {
      size: "sm",
      variant: "outline",
      onclick: () => graph.resetRun(),
      disabled: graph.running,
      children: ($$renderer3) => {
        $$renderer3.push(`<!---->Clear run`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> `);
    Button($$renderer2, {
      size: "sm",
      variant: "ghost",
      onclick: () => graph.reset(),
      disabled: graph.running,
      children: ($$renderer3) => {
        Rotate_ccw($$renderer3, { class: "size-3.5" });
        $$renderer3.push(`<!----> Reset`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> `);
    Button($$renderer2, {
      size: "sm",
      variant: "outline",
      onclick: () => graph.deleteSelected(),
      disabled: !graph.hasSelection || graph.running,
      title: "Delete the selected node(s) / edge(s)",
      children: ($$renderer3) => {
        Trash_2($$renderer3, { class: "size-3.5" });
        $$renderer3.push(`<!----> Delete`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----></div> <div class="flex items-center gap-1.5">`);
    Button($$renderer2, {
      size: "sm",
      variant: "outline",
      onclick: () => graph.undo(),
      disabled: !graph.canUndo || graph.running,
      title: "Undo (Ctrl/Cmd+Z)",
      children: ($$renderer3) => {
        Undo_2($$renderer3, { class: "size-3.5" });
        $$renderer3.push(`<!----> Undo`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> `);
    Button($$renderer2, {
      size: "sm",
      variant: "outline",
      onclick: () => graph.redo(),
      disabled: !graph.canRedo || graph.running,
      title: "Redo (Ctrl/Cmd+Shift+Z)",
      children: ($$renderer3) => {
        Redo_2($$renderer3, { class: "size-3.5" });
        $$renderer3.push(`<!----> Redo`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> `);
    Button($$renderer2, {
      size: "sm",
      variant: "ghost",
      onclick: () => graph.tidy(),
      disabled: graph.running,
      title: "Auto-layout the graph left-to-right",
      children: ($$renderer3) => {
        Wand_sparkles($$renderer3, { class: "size-3.5" });
        $$renderer3.push(`<!----> Tidy`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> `);
    Button($$renderer2, {
      size: "sm",
      variant: "ghost",
      onclick: () => commandMenu.toggle(),
      title: "All commands & shortcuts (Ctrl/⌘ K)",
      children: ($$renderer3) => {
        Command$1($$renderer3, { class: "size-3.5" });
        $$renderer3.push(`<!----> ⌘K`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----></div> `);
    if (graph.lastError) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="max-w-[18rem] text-[10px] text-destructive">${escape_html(graph.lastError)}</div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> <div class="max-w-[19rem] space-y-0.5 text-[10px] text-muted-foreground/80"><div><span class="text-foreground">Add</span> — drag a node from the palette (top-right) onto the canvas.</div> <div><span class="text-foreground">Connect</span> — drag a node's right ● onto another's left ●. A Search
      accepts several inputs at once (a query/image + a refine).</div> <div><span class="text-foreground">Run one node</span> — hover a node and press ▶: upstream results are
      reused, missing upstream runs once. Shift+▶ reruns the whole branch; an amber “stale” chip means
      upstream changed since that node last ran.</div> <div><span class="text-foreground">Delete</span> — hover a node and click ✕, or select a node/edge and
      press ⌫ (or the Delete button).</div> <div><span class="text-foreground">Refine</span> — Search → Search scopes the second to the first's videos.
      Click any node to inspect it on the right.</div> <div><span class="text-foreground">Export</span> — wire results into an Export node, then pick the format
      &amp; columns in the inspector and download.</div></div></div>`);
  });
}
function NodePalette($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<div class="flex max-w-[8.5rem] flex-col gap-1 rounded-lg border border-border bg-card/95 p-1.5 shadow-md backdrop-blur"><span class="px-1 text-[10px] font-medium tracking-wide text-muted-foreground uppercase">Drag to add</span> <!--[-->`);
    const each_array = ensure_array_like(NODE_KINDS);
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let kind = each_array[$$index];
      $$renderer2.push(`<button type="button" draggable="true"${attr("title", `Drag onto the canvas to add a ${stringify(nodeLabel(kind))} node`)} class="flex cursor-grab items-center gap-1.5 rounded border border-border bg-background px-2 py-1 text-[11px] text-foreground transition-colors hover:bg-muted active:cursor-grabbing">`);
      Grip_vertical($$renderer2, { class: "size-3 shrink-0 text-muted-foreground" });
      $$renderer2.push(`<!----> <span class="truncate">${escape_html(nodeLabel(kind))}</span></button>`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function ContextMenu($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { menu } = $$props;
    const cfg = derived(() => menu.nodeId ? graph.config[menu.nodeId] : null);
    const ITEM = "flex w-full items-center gap-2 rounded px-2 py-1 text-left transition-colors hover:bg-muted";
    $$renderer2.push(`<div class="fixed inset-0 z-40" role="presentation"></div> <div class="fixed z-50 min-w-[9rem] rounded-md border border-border bg-card p-1 text-xs text-foreground shadow-lg"${attr_style(`left: ${stringify(menu.x)}px; top: ${stringify(menu.y)}px;`)} role="menu" tabindex="-1">`);
    if (menu.mode === "node" && menu.nodeId && cfg()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<button${attr_class(clsx(ITEM))} role="menuitem"${attr("disabled", graph.running, true)}>`);
      Play($$renderer2, { class: "size-3.5" });
      $$renderer2.push(`<!----> Run node</button> <button${attr_class(clsx(ITEM))} role="menuitem"${attr("disabled", graph.running, true)} title="Re-execute this node AND everything upstream of it">`);
      Refresh_cw($$renderer2, { class: "size-3.5" });
      $$renderer2.push(`<!----> Run branch fresh</button> <button${attr_class(clsx(ITEM))} role="menuitem">`);
      Copy($$renderer2, { class: "size-3.5" });
      $$renderer2.push(`<!----> Duplicate</button> <button${attr_class(clsx(ITEM))} role="menuitem"${attr("disabled", graph.running, true)}>`);
      if (cfg().enabled) {
        $$renderer2.push("<!--[0-->");
        Eye_off($$renderer2, { class: "size-3.5" });
        $$renderer2.push(`<!----> Disable`);
      } else {
        $$renderer2.push("<!--[-1-->");
        Eye($$renderer2, { class: "size-3.5" });
        $$renderer2.push(`<!----> Enable`);
      }
      $$renderer2.push(`<!--]--></button> <button${attr_class(`${stringify(ITEM)} text-destructive`)} role="menuitem">`);
      Trash_2($$renderer2, { class: "size-3.5" });
      $$renderer2.push(`<!----> Delete</button>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div class="px-2 py-1 text-[10px] tracking-wide text-muted-foreground uppercase">Add node</div> <!--[-->`);
      const each_array = ensure_array_like(NODE_KINDS);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let kind = each_array[$$index];
        $$renderer2.push(`<button${attr_class(clsx(ITEM))} role="menuitem">`);
        Plus$1($$renderer2, { class: "size-3.5" });
        $$renderer2.push(`<!----> ${escape_html(nodeLabel(kind))}</button>`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function FlowPane($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { screenToFlowPosition } = useSvelteFlow();
    const edgeTypes = { default: ReconnectableEdge };
    const INVALID_TOAST_MS = 2600;
    const SNAP_GRID_PX = 16;
    const theme = useColorMode();
    let invalidMsg = null;
    let pendingReason = null;
    let madeConnection = false;
    let msgTimer = null;
    function flash(msg) {
      invalidMsg = msg;
      if (msgTimer) clearTimeout(msgTimer);
      msgTimer = setTimeout(() => invalidMsg = null, INVALID_TOAST_MS);
    }
    function validate(connection) {
      const reason = graph.connectionError(connection);
      if (reason && connection.source && connection.target) pendingReason = reason;
      return reason === null;
    }
    let menu = null;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="h-full w-full">`);
      SvelteFlow($$renderer3, {
        nodeTypes,
        edgeTypes,
        colorMode: theme.current,
        fitView: true,
        deleteKey: ["Backspace", "Delete"],
        snapGrid: [SNAP_GRID_PX, SNAP_GRID_PX],
        defaultEdgeOptions: { markerEnd: ARROW_MARKER },
        isValidConnection: validate,
        onnodeclick: (e) => graph.inspectNode(e.node.id),
        onnodecontextmenu: ({ node, event }) => {
          event.preventDefault();
          menu = {
            x: event.clientX,
            y: event.clientY,
            mode: "node",
            nodeId: node.id,
            flow: { x: 0, y: 0 }
          };
        },
        onpanecontextmenu: ({ event }) => {
          event.preventDefault();
          menu = {
            x: event.clientX,
            y: event.clientY,
            mode: "pane",
            nodeId: null,
            flow: screenToFlowPosition({ x: event.clientX, y: event.clientY })
          };
        },
        onconnectstart: () => {
          pendingReason = null;
          madeConnection = false;
          invalidMsg = null;
        },
        onconnect: () => {
          madeConnection = true;
          pendingReason = null;
        },
        onconnectend: (_event, connectionState) => {
          if (!madeConnection && pendingReason && connectionState?.toHandle) flash(pendingReason);
        },
        onbeforedelete: async ({ nodes }) => {
          const feedsOthers = nodes.some((n2) => graph.dependentsOf(n2.id).length > 0);
          return !feedsOthers || window.confirm("Delete node(s) that feed others downstream?");
        },
        ondelete: ({ nodes, edges }) => graph.syncDeleted(nodes.map((n2) => n2.id), edges.map((e) => e.id)),
        onselectionchange: (p2) => graph.setSelection(p2.nodes.map((n2) => n2.id), p2.edges.map((e) => e.id)),
        get nodes() {
          return graph.nodes;
        },
        set nodes($$value) {
          graph.nodes = $$value;
          $$settled = false;
        },
        get edges() {
          return graph.edges;
        },
        set edges($$value) {
          graph.edges = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          Background($$renderer4, {});
          $$renderer4.push(`<!----> `);
          Controls($$renderer4, {});
          $$renderer4.push(`<!----> `);
          Minimap($$renderer4, {});
          $$renderer4.push(`<!----> `);
          Panel($$renderer4, {
            position: "top-left",
            children: ($$renderer5) => {
              WorkflowToolbar($$renderer5);
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          Panel($$renderer4, {
            position: "top-right",
            children: ($$renderer5) => {
              NodePalette($$renderer5);
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          if (invalidMsg) {
            $$renderer4.push("<!--[0-->");
            Panel($$renderer4, {
              position: "bottom-center",
              children: ($$renderer5) => {
                $$renderer5.push(`<div class="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-1.5 text-xs font-medium text-destructive shadow">${escape_html(invalidMsg)}</div>`);
              },
              $$slots: { default: true }
            });
          } else {
            $$renderer4.push("<!--[-1-->");
          }
          $$renderer4.push(`<!--]-->`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----></div> `);
      if (menu) {
        $$renderer3.push("<!--[0-->");
        ContextMenu($$renderer3, { menu });
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
function WorkflowCanvas($$renderer) {
  $$renderer.push(`<div class="h-full w-full">`);
  SvelteFlowProvider($$renderer, {
    children: ($$renderer2) => {
      FlowPane($$renderer2);
    }
  });
  $$renderer.push(`<!----></div>`);
}
function WorkflowInspector($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const id = derived(() => graph.inspectedNodeId);
    const kind = derived(() => id() ? graph.kindOf(id()) : null);
    const cfg = derived(() => id() ? graph.config[id()] : null);
    const rt = derived(() => id() ? graph.runtime[id()] : null);
    const hits = derived(() => rt()?.hits ?? []);
    const EXPORT_FORMATS = ["csv", "json"];
    const allColumns = derived(exportColumns);
    const selectedColumns = derived(() => cfg() ? cfg().exportColumns ?? allColumns() : []);
    const title = derived(() => kind() === "search" && cfg() ? `Search · ${modeLabel(cfg().mode)}` : kind() ? nodeLabel(kind()) : "");
    const statusText = derived(() => {
      if (!rt()) return "";
      if (rt().status === "running") return "searching…";
      if (rt().status === "error") return "error";
      if (rt().status === "done") return rt().count != null ? `done · ${rt().count} hits` : "done";
      return "not run yet";
    });
    const rows = derived(() => {
      if (!cfg() || !kind()) return [];
      const r = [];
      if (kind() === "query") r.push(["Query", cfg().q || "—"]);
      if (kind() === "image") r.push(["Image", cfg().imageName || "(none uploaded)"]);
      if (kind() === "filter") {
        if (cfg().where) r.push(["Where", cfg().where]);
        const view = activeView();
        for (const field of view.filterFields) {
          const value = cfg().filters[field];
          if (value) {
            const label = view.metadataFields.find((m) => m.field === field)?.label ?? field;
            r.push([label, value]);
          }
        }
        if (!r.length) r.push(["Filter", "(empty)"]);
      }
      if (kind() === "search") {
        r.push(["Mode", modeLabel(cfg().mode)]);
        r.push([
          "Query",
          cfg().q || (cfg().mode === "visual" ? "(from image)" : "—")
        ]);
        r.push(["Results", String(cfg().n)]);
        if (cfg().rerank) r.push(["Rerank", `top ${RERANK_TOP_N}`]);
        if (rt()?.scopedDocs) r.push([
          "Scope",
          `within ${rt().scopedDocs} videos${rt().scopeCapped ? " (capped)" : ""}`
        ]);
        if (rt()?.scopedChunks) r.push([
          "Scope",
          `within ${rt().scopedChunks} chunks${rt().scopeCapped ? " (capped)" : ""}`
        ]);
        if (rt()?.ms != null) r.push(["Time", `${rt().ms} ms`]);
      }
      if (kind() === "combine") {
        r.push([
          "Combine",
          cfg().combineMode === "intersect" ? "intersect (∩)" : "union (∪)"
        ]);
      }
      if (kind() === "tagger") {
        r.push([
          "Tags",
          cfg().tags.length ? cfg().tags.join(", ") : "(none — add some)"
        ]);
      }
      return r;
    });
    $$renderer2.push(`<div data-testid="inspector" class="flex h-full min-h-0 flex-col border-l border-border bg-card"><header class="flex h-11 shrink-0 items-center gap-2 border-b border-border px-3">`);
    if (graph.selectedHit) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<button type="button" aria-label="Back to results" class="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">`);
      Arrow_left($$renderer2, { class: "size-4" });
      $$renderer2.push(`<!----></button> <span class="truncate text-sm font-medium text-foreground">${escape_html(activeView().title(graph.selectedHit) || "Now playing")}</span>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<span class="text-sm font-medium text-foreground">Inspector</span> `);
      if (title()) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<span class="truncate text-xs text-muted-foreground">· ${escape_html(title())}</span>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></header> <div class="min-h-0 flex-1 overflow-y-auto">`);
    if (graph.selectedHit) {
      $$renderer2.push("<!--[0-->");
      Player_pane($$renderer2, { hit: graph.selectedHit });
    } else if (id() && kind() && cfg() && rt()) {
      $$renderer2.push("<!--[1-->");
      $$renderer2.push(`<div class="flex flex-col gap-3 p-3 text-xs"><div class="flex items-center gap-1.5"><input class="min-w-0 flex-1 rounded border border-border bg-background px-2 py-1 text-xs text-foreground outline-none focus:border-primary"${attr("placeholder", title())} aria-label="Rename node"${attr("value", cfg().label)}/> <button type="button" title="Duplicate node" aria-label="Duplicate node" class="shrink-0 rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">`);
      Copy($$renderer2, { class: "size-3.5" });
      $$renderer2.push(`<!----></button> <button type="button"${attr("title", cfg().enabled ? "Disable (bypass) node" : "Enable node")} aria-label="Toggle node enabled" class="shrink-0 rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">`);
      if (cfg().enabled) {
        $$renderer2.push("<!--[0-->");
        Eye_off($$renderer2, { class: "size-3.5" });
      } else {
        $$renderer2.push("<!--[-1-->");
        Eye($$renderer2, { class: "size-3.5" });
      }
      $$renderer2.push(`<!--]--></button></div> <div class="flex items-center gap-2"><span${attr_class(`size-2 shrink-0 rounded-full ${stringify(STATUS_DOT[rt().status])}`)}></span> <span class="text-muted-foreground">${escape_html(statusText())}</span></div> `);
      if (rt().error) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="rounded border border-destructive/30 bg-destructive/10 p-2 text-destructive">${escape_html(rt().error)}</div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1"><!--[-->`);
      const each_array = ensure_array_like(rows());
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let [label, value] = each_array[$$index];
        $$renderer2.push(`<dt class="text-muted-foreground">${escape_html(label)}</dt> <dd class="break-words font-medium text-foreground">${escape_html(value)}</dd>`);
      }
      $$renderer2.push(`<!--]--></dl> `);
      if (kind() === "export") {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="flex flex-col gap-2.5 border-t border-border pt-3"><div class="flex items-center gap-2"><span class="text-muted-foreground">Format</span> <div class="flex overflow-hidden rounded border border-border"><!--[-->`);
        const each_array_1 = ensure_array_like(EXPORT_FORMATS);
        for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
          let fmt = each_array_1[$$index_1];
          $$renderer2.push(`<button type="button"${attr_class(`px-2.5 py-0.5 text-[11px] font-medium transition-colors ${stringify(cfg().exportFormat === fmt ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted")}`)}>${escape_html(fmt.toUpperCase())}</button>`);
        }
        $$renderer2.push(`<!--]--></div></div> <div><div class="mb-1 flex items-center justify-between"><span class="text-[10px] tracking-wide text-muted-foreground uppercase">Columns (${escape_html(selectedColumns().length)}/${escape_html(allColumns().length)})</span> <div class="flex gap-2 text-[10px]"><button type="button" class="text-primary hover:underline">All</button> <button type="button" class="text-primary hover:underline">None</button></div></div> <div class="grid grid-cols-2 gap-x-3 gap-y-0.5"><!--[-->`);
        const each_array_2 = ensure_array_like(allColumns());
        for (let $$index_2 = 0, $$length = each_array_2.length; $$index_2 < $$length; $$index_2++) {
          let col = each_array_2[$$index_2];
          $$renderer2.push(`<label class="flex items-center gap-1.5 text-[11px] text-foreground"><input type="checkbox" class="size-3 accent-primary"${attr("checked", selectedColumns().includes(col), true)}/> <span class="truncate"${attr("title", col)}>${escape_html(col)}</span></label>`);
        }
        $$renderer2.push(`<!--]--></div></div> <button type="button" class="inline-flex items-center justify-center gap-1.5 rounded border border-border bg-background px-2 py-1.5 text-[11px] font-medium text-foreground transition-colors hover:bg-muted disabled:opacity-50"${attr("disabled", hits().length === 0 || selectedColumns().length === 0, true)}>`);
        Download($$renderer2, { class: "size-3.5" });
        $$renderer2.push(`<!----> Download ${escape_html(hits().length)} hit${escape_html(hits().length === 1 ? "" : "s")} as ${escape_html(cfg().exportFormat.toUpperCase())}</button></div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (hits().length) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div><div class="mb-1 text-[10px] tracking-wide text-muted-foreground uppercase">Results (${escape_html(hits().length)}) · click to play</div> `);
        HitList($$renderer2, { hits: hits(), maxHeight: "max-h-none" });
        $$renderer2.push(`<!----></div>`);
      } else if (kind() === "search" || kind() === "results" || kind() === "export" || kind() === "combine" || kind() === "tagger") {
        $$renderer2.push("<!--[1-->");
        $$renderer2.push(`<p class="text-[11px] text-muted-foreground">`);
        if (kind() === "export" && rt().status === "idle") {
          $$renderer2.push("<!--[0-->");
          $$renderer2.push(`Press Run to feed results. Selected columns${escape_html(selectedColumns().includes("tags") ? " (including tags)" : "")} will export.`);
        } else if (rt().status === "idle") {
          $$renderer2.push("<!--[1-->");
          $$renderer2.push(`Not run yet — press Run.`);
        } else {
          $$renderer2.push("<!--[-1-->");
          $$renderer2.push(`No results.`);
        }
        $$renderer2.push(`<!--]--></p>`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<p class="text-[11px] text-muted-foreground">Produces a ${escape_html(nodeLabel(kind()).toLowerCase())} input — wire it into a Search and Run.</p>`);
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div class="grid h-full place-items-center p-6 text-center text-xs text-muted-foreground">Click a node to inspect its inputs &amp; results — or click a result to play it.</div>`);
    }
    $$renderer2.push(`<!--]--></div></div>`);
  });
}
function CommandMenu($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const selectedId = derived(() => graph.selectedNodeIds.length === 1 ? graph.selectedNodeIds[0] ?? null : null);
    const selectedTitle = derived(() => {
      if (!selectedId()) return null;
      const kind = graph.kindOf(selectedId());
      return graph.config[selectedId()]?.label?.trim() || (kind ? nodeLabel(kind) : selectedId());
    });
    function act(fn) {
      commandMenu.open = false;
      fn();
    }
    const ITEM = "flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm outline-none data-selected:bg-secondary data-disabled:cursor-default data-disabled:opacity-40";
    const CHIP = "ml-auto shrink-0 rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground";
    const HEADING = "px-2 pt-2 pb-1 text-[10px] tracking-wide text-muted-foreground uppercase";
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (Dialog) {
        $$renderer3.push("<!--[-->");
        Dialog($$renderer3, {
          get open() {
            return commandMenu.open;
          },
          set open($$value) {
            commandMenu.open = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            if (Portal) {
              $$renderer4.push("<!--[-->");
              Portal($$renderer4, {
                children: ($$renderer5) => {
                  if (Dialog_overlay) {
                    $$renderer5.push("<!--[-->");
                    Dialog_overlay($$renderer5, { class: "fixed inset-0 z-50 bg-black/40" });
                    $$renderer5.push("<!--]-->");
                  } else {
                    $$renderer5.push("<!--[!-->");
                    $$renderer5.push("<!--]-->");
                  }
                  $$renderer5.push(` `);
                  if (Dialog_content) {
                    $$renderer5.push("<!--[-->");
                    Dialog_content($$renderer5, {
                      class: "fixed top-[20%] left-1/2 z-50 w-[34rem] max-w-[calc(100vw-2rem)] -translate-x-1/2 overflow-hidden rounded-lg border border-border bg-card shadow-xl",
                      children: ($$renderer6) => {
                        if (Dialog_title) {
                          $$renderer6.push("<!--[-->");
                          Dialog_title($$renderer6, {
                            class: "sr-only",
                            children: ($$renderer7) => {
                              $$renderer7.push(`<!---->Workflow commands`);
                            },
                            $$slots: { default: true }
                          });
                          $$renderer6.push("<!--]-->");
                        } else {
                          $$renderer6.push("<!--[!-->");
                          $$renderer6.push("<!--]-->");
                        }
                        $$renderer6.push(` `);
                        if (Command) {
                          $$renderer6.push("<!--[-->");
                          Command($$renderer6, {
                            class: "flex flex-col",
                            children: ($$renderer7) => {
                              if (Command_input) {
                                $$renderer7.push("<!--[-->");
                                Command_input($$renderer7, {
                                  placeholder: "Type a command…",
                                  class: "w-full border-b border-border bg-transparent px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                                });
                                $$renderer7.push("<!--]-->");
                              } else {
                                $$renderer7.push("<!--[!-->");
                                $$renderer7.push("<!--]-->");
                              }
                              $$renderer7.push(` `);
                              if (Command_list) {
                                $$renderer7.push("<!--[-->");
                                Command_list($$renderer7, {
                                  class: "max-h-[22rem] overflow-y-auto p-1.5",
                                  children: ($$renderer8) => {
                                    if (Command_viewport) {
                                      $$renderer8.push("<!--[-->");
                                      Command_viewport($$renderer8, {
                                        children: ($$renderer9) => {
                                          if (Command_empty) {
                                            $$renderer9.push("<!--[-->");
                                            Command_empty($$renderer9, {
                                              class: "px-2 py-6 text-center text-xs text-muted-foreground",
                                              children: ($$renderer10) => {
                                                $$renderer10.push(`<!---->No matching command.`);
                                              },
                                              $$slots: { default: true }
                                            });
                                            $$renderer9.push("<!--]-->");
                                          } else {
                                            $$renderer9.push("<!--[!-->");
                                            $$renderer9.push("<!--]-->");
                                          }
                                          $$renderer9.push(` `);
                                          if (Command_group) {
                                            $$renderer9.push("<!--[-->");
                                            Command_group($$renderer9, {
                                              children: ($$renderer10) => {
                                                if (Command_group_heading) {
                                                  $$renderer10.push("<!--[-->");
                                                  Command_group_heading($$renderer10, {
                                                    class: HEADING,
                                                    children: ($$renderer11) => {
                                                      $$renderer11.push(`<!---->Run`);
                                                    },
                                                    $$slots: { default: true }
                                                  });
                                                  $$renderer10.push("<!--]-->");
                                                } else {
                                                  $$renderer10.push("<!--[!-->");
                                                  $$renderer10.push("<!--]-->");
                                                }
                                                $$renderer10.push(` `);
                                                if (Command_group_items) {
                                                  $$renderer10.push("<!--[-->");
                                                  Command_group_items($$renderer10, {
                                                    children: ($$renderer11) => {
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "run workflow all",
                                                          disabled: graph.running,
                                                          onSelect: () => act(() => void graph.run()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Play($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Run workflow (all nodes, fresh)`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "run selected node",
                                                          keywords: ["play", "execute", "partial"],
                                                          disabled: graph.running || !selectedId(),
                                                          onSelect: () => act(() => void graph.runNode(selectedId())),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Play($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Run selected node${escape_html(selectedTitle() ? ` — ${selectedTitle()}` : "")} <span${attr_class(clsx(CHIP))}>▶ on node</span>`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "run selected branch fresh",
                                                          keywords: ["rerun", "upstream", "force"],
                                                          disabled: graph.running || !selectedId(),
                                                          onSelect: () => act(() => void graph.runNode(selectedId(), { fresh: true })),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Refresh_cw($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Run selected branch fresh${escape_html(selectedTitle() ? ` — ${selectedTitle()}` : "")} <span${attr_class(clsx(CHIP))}>Shift+▶</span>`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "clear run results",
                                                          disabled: graph.running,
                                                          onSelect: () => act(() => graph.resetRun()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Eraser($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Clear run results`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                    },
                                                    $$slots: { default: true }
                                                  });
                                                  $$renderer10.push("<!--]-->");
                                                } else {
                                                  $$renderer10.push("<!--[!-->");
                                                  $$renderer10.push("<!--]-->");
                                                }
                                              },
                                              $$slots: { default: true }
                                            });
                                            $$renderer9.push("<!--]-->");
                                          } else {
                                            $$renderer9.push("<!--[!-->");
                                            $$renderer9.push("<!--]-->");
                                          }
                                          $$renderer9.push(` `);
                                          if (Command_separator) {
                                            $$renderer9.push("<!--[-->");
                                            Command_separator($$renderer9, { class: "my-1 h-px bg-border" });
                                            $$renderer9.push("<!--]-->");
                                          } else {
                                            $$renderer9.push("<!--[!-->");
                                            $$renderer9.push("<!--]-->");
                                          }
                                          $$renderer9.push(` `);
                                          if (Command_group) {
                                            $$renderer9.push("<!--[-->");
                                            Command_group($$renderer9, {
                                              children: ($$renderer10) => {
                                                if (Command_group_heading) {
                                                  $$renderer10.push("<!--[-->");
                                                  Command_group_heading($$renderer10, {
                                                    class: HEADING,
                                                    children: ($$renderer11) => {
                                                      $$renderer11.push(`<!---->Edit`);
                                                    },
                                                    $$slots: { default: true }
                                                  });
                                                  $$renderer10.push("<!--]-->");
                                                } else {
                                                  $$renderer10.push("<!--[!-->");
                                                  $$renderer10.push("<!--]-->");
                                                }
                                                $$renderer10.push(` `);
                                                if (Command_group_items) {
                                                  $$renderer10.push("<!--[-->");
                                                  Command_group_items($$renderer10, {
                                                    children: ($$renderer11) => {
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "undo",
                                                          disabled: !graph.canUndo || graph.running,
                                                          onSelect: () => act(() => graph.undo()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Undo_2($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Undo <span${attr_class(clsx(CHIP))}>Ctrl/⌘ Z</span>`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "redo",
                                                          disabled: !graph.canRedo || graph.running,
                                                          onSelect: () => act(() => graph.redo()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Redo_2($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Redo <span${attr_class(clsx(CHIP))}>Ctrl/⌘ ⇧ Z</span>`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "copy selection",
                                                          disabled: !graph.hasSelection,
                                                          onSelect: () => act(() => graph.copySelection()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Copy($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Copy selection <span${attr_class(clsx(CHIP))}>Ctrl/⌘ C</span>`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "paste",
                                                          disabled: graph.running,
                                                          onSelect: () => act(() => graph.paste()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Clipboard_paste($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Paste <span${attr_class(clsx(CHIP))}>Ctrl/⌘ V</span>`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "delete selection",
                                                          disabled: !graph.hasSelection || graph.running,
                                                          onSelect: () => act(() => graph.deleteSelected()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Trash_2($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Delete selection <span${attr_class(clsx(CHIP))}>⌫</span>`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "tidy auto layout",
                                                          disabled: graph.running,
                                                          onSelect: () => act(() => graph.tidy()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Wand_sparkles($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Tidy (auto-layout)`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "reset graph to starter",
                                                          disabled: graph.running,
                                                          onSelect: () => act(() => graph.reset()),
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            Rotate_ccw($$renderer12, { class: "size-3.5" });
                                                            $$renderer12.push(`<!----> Reset to starter graph`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                    },
                                                    $$slots: { default: true }
                                                  });
                                                  $$renderer10.push("<!--]-->");
                                                } else {
                                                  $$renderer10.push("<!--[!-->");
                                                  $$renderer10.push("<!--]-->");
                                                }
                                              },
                                              $$slots: { default: true }
                                            });
                                            $$renderer9.push("<!--]-->");
                                          } else {
                                            $$renderer9.push("<!--[!-->");
                                            $$renderer9.push("<!--]-->");
                                          }
                                          $$renderer9.push(` `);
                                          if (Command_separator) {
                                            $$renderer9.push("<!--[-->");
                                            Command_separator($$renderer9, { class: "my-1 h-px bg-border" });
                                            $$renderer9.push("<!--]-->");
                                          } else {
                                            $$renderer9.push("<!--[!-->");
                                            $$renderer9.push("<!--]-->");
                                          }
                                          $$renderer9.push(` `);
                                          if (Command_group) {
                                            $$renderer9.push("<!--[-->");
                                            Command_group($$renderer9, {
                                              children: ($$renderer10) => {
                                                if (Command_group_heading) {
                                                  $$renderer10.push("<!--[-->");
                                                  Command_group_heading($$renderer10, {
                                                    class: HEADING,
                                                    children: ($$renderer11) => {
                                                      $$renderer11.push(`<!---->On a node card`);
                                                    },
                                                    $$slots: { default: true }
                                                  });
                                                  $$renderer10.push("<!--]-->");
                                                } else {
                                                  $$renderer10.push("<!--[!-->");
                                                  $$renderer10.push("<!--]-->");
                                                }
                                                $$renderer10.push(` `);
                                                if (Command_group_items) {
                                                  $$renderer10.push("<!--[-->");
                                                  Command_group_items($$renderer10, {
                                                    children: ($$renderer11) => {
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "hint play node",
                                                          disabled: true,
                                                          forceMount: true,
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            $$renderer12.push(`<!---->▶ — run that node; upstream results are reused, missing upstream runs once`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "hint shift play branch",
                                                          disabled: true,
                                                          forceMount: true,
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            $$renderer12.push(`<!---->Shift+▶ — re-execute the node and its whole upstream branch`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "hint stale badge",
                                                          disabled: true,
                                                          forceMount: true,
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            $$renderer12.push(`<!---->amber “stale” — upstream re-ran since this node's results; ▶ refreshes`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                      $$renderer11.push(` `);
                                                      if (Command_item) {
                                                        $$renderer11.push("<!--[-->");
                                                        Command_item($$renderer11, {
                                                          value: "hint right click menu",
                                                          disabled: true,
                                                          forceMount: true,
                                                          class: ITEM,
                                                          children: ($$renderer12) => {
                                                            $$renderer12.push(`<!---->right-click — run / duplicate / disable / delete; right-click canvas adds a node`);
                                                          },
                                                          $$slots: { default: true }
                                                        });
                                                        $$renderer11.push("<!--]-->");
                                                      } else {
                                                        $$renderer11.push("<!--[!-->");
                                                        $$renderer11.push("<!--]-->");
                                                      }
                                                    },
                                                    $$slots: { default: true }
                                                  });
                                                  $$renderer10.push("<!--]-->");
                                                } else {
                                                  $$renderer10.push("<!--[!-->");
                                                  $$renderer10.push("<!--]-->");
                                                }
                                              },
                                              $$slots: { default: true }
                                            });
                                            $$renderer9.push("<!--]-->");
                                          } else {
                                            $$renderer9.push("<!--[!-->");
                                            $$renderer9.push("<!--]-->");
                                          }
                                        },
                                        $$slots: { default: true }
                                      });
                                      $$renderer8.push("<!--]-->");
                                    } else {
                                      $$renderer8.push("<!--[!-->");
                                      $$renderer8.push("<!--]-->");
                                    }
                                  },
                                  $$slots: { default: true }
                                });
                                $$renderer7.push("<!--]-->");
                              } else {
                                $$renderer7.push("<!--[!-->");
                                $$renderer7.push("<!--]-->");
                              }
                            },
                            $$slots: { default: true }
                          });
                          $$renderer6.push("<!--]-->");
                        } else {
                          $$renderer6.push("<!--[!-->");
                          $$renderer6.push("<!--]-->");
                        }
                      },
                      $$slots: { default: true }
                    });
                    $$renderer5.push("<!--]-->");
                  } else {
                    $$renderer5.push("<!--[!-->");
                    $$renderer5.push("<!--]-->");
                  }
                }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
          },
          $$slots: { default: true }
        });
        $$renderer3.push("<!--]-->");
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push("<!--]-->");
      }
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    CommandMenu($$renderer2);
    $$renderer2.push(`<!----> <div class="h-full min-h-0 w-full">`);
    {
      let left = function($$renderer3) {
        WorkflowCanvas($$renderer3);
      }, right = function($$renderer3) {
        WorkflowInspector($$renderer3);
      };
      Resizable_split($$renderer2, {
        storageKey: "lance-media-workflow-split",
        initial: 0.7,
        minLeft: 460,
        minRight: 300,
        left,
        right
      });
    }
    $$renderer2.push(`<!----></div>`);
  });
}
export {
  _page as default
};
