const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('screenStudio', {
  listSources: () => ipcRenderer.invoke('sources:list'),
  selectSource: (id) => ipcRenderer.invoke('sources:select', id),
  saveRecording: (arrayBuffer) => ipcRenderer.invoke('recording:save', arrayBuffer)
});
