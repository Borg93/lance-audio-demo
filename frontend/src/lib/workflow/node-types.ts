/** Maps each node `type` to its Svelte component. Passed to <SvelteFlow>. */
import type { NodeTypes } from '@xyflow/svelte';
import QueryNode from './nodes/QueryNode.svelte';
import ImageNode from './nodes/ImageNode.svelte';
import FilterNode from './nodes/FilterNode.svelte';
import SearchNode from './nodes/SearchNode.svelte';
import CombineNode from './nodes/CombineNode.svelte';
import ResultsNode from './nodes/ResultsNode.svelte';

export const nodeTypes: NodeTypes = {
  query: QueryNode,
  image: ImageNode,
  filter: FilterNode,
  search: SearchNode,
  combine: CombineNode,
  results: ResultsNode,
};
