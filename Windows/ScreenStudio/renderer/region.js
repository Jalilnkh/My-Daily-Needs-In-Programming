const selection = document.querySelector('#selection');
const size = document.querySelector('#size');
let start = null;

window.addEventListener('pointerdown', event => {
  start = { x: event.clientX, y: event.clientY };
  selection.style.display = 'block';
  update(event);
});

window.addEventListener('pointermove', event => start && update(event));
window.addEventListener('pointerup', event => {
  if (!start) return;
  const rect = rectangle(event);
  start = null;
  if (rect.width < 10 || rect.height < 10) return;
  window.regionSelector.finish({
    x: rect.left / innerWidth,
    y: rect.top / innerHeight,
    w: rect.width / innerWidth,
    h: rect.height / innerHeight
  });
});
window.addEventListener('keydown', event => event.key === 'Escape' && window.regionSelector.finish(null));

function rectangle(event) {
  const left = Math.max(0, Math.min(start.x, event.clientX));
  const top = Math.max(0, Math.min(start.y, event.clientY));
  const right = Math.min(innerWidth, Math.max(start.x, event.clientX));
  const bottom = Math.min(innerHeight, Math.max(start.y, event.clientY));
  return { left, top, width: right - left, height: bottom - top };
}

function update(event) {
  const rect = rectangle(event);
  Object.assign(selection.style, { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` });
  size.textContent = `${Math.round(rect.width * devicePixelRatio)} × ${Math.round(rect.height * devicePixelRatio)}`;
}
