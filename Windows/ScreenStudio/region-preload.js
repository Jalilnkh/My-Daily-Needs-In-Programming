const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('regionSelector', {
  finish: region => ipcRenderer.send('region:selected', region)
});
