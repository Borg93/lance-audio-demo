let async_mode_flag = false;
let tracing_mode_flag = false;
function enable_async_mode_flag() {
  async_mode_flag = true;
}
export {
  async_mode_flag as a,
  enable_async_mode_flag as e,
  tracing_mode_flag as t
};
