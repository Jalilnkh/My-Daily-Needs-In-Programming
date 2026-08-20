const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('screenStudio', {
  listSources: () => ipcRenderer.invoke('sources:list'),
  selectSource: (id) => ipcRenderer.invoke('sources:select', id),
  selectRegion: (sourceId) => ipcRenderer.invoke('region:select', sourceId),
  startRecordingFile: () => ipcRenderer.invoke('recording:start'),
  appendRecordingChunk: (id, arrayBuffer) => ipcRenderer.invoke('recording:append', id, arrayBuffer),
  finishRecordingFile: (id) => ipcRenderer.invoke('recording:finish', id),
  abortRecordingFile: (id) => ipcRenderer.invoke('recording:abort', id)
});
