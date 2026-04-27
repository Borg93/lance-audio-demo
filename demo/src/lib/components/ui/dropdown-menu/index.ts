import { DropdownMenu as DropdownMenuPrimitive } from 'bits-ui';

import Trigger from './dropdown-menu-trigger.svelte';
import Content from './dropdown-menu-content.svelte';
import Item from './dropdown-menu-item.svelte';
import RadioGroup from './dropdown-menu-radio-group.svelte';
import RadioItem from './dropdown-menu-radio-item.svelte';
import Label from './dropdown-menu-label.svelte';
import Separator from './dropdown-menu-separator.svelte';

const Root = DropdownMenuPrimitive.Root;
const Group = DropdownMenuPrimitive.Group;
const Portal = DropdownMenuPrimitive.Portal;

export {
	Root,
	Trigger,
	Content,
	Item,
	RadioGroup,
	RadioItem,
	Label,
	Separator,
	Group,
	Portal,
	//
	Root as DropdownMenu,
	Trigger as DropdownMenuTrigger,
	Content as DropdownMenuContent,
	Item as DropdownMenuItem,
	RadioGroup as DropdownMenuRadioGroup,
	RadioItem as DropdownMenuRadioItem,
	Label as DropdownMenuLabel,
	Separator as DropdownMenuSeparator
};
