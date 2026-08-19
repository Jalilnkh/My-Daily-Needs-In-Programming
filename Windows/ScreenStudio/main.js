const { app, BrowserWindow, desktopCapturer, dialog, ipcMain, session } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');

let selectedSourceId = null;

function createWindow() {
  const win = new BrowserWindow({
    width: 1240,
    height: 790,
    minWidth: 980,
    minHeight: 680,
    backgroundColor: '#090b10',
    title: 'Screen Studio',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.removeMenu();
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  session.defaultSession.setDisplayMediaRequestHandler(async (_request, callback) => {
    const sources = await desktopCapturer.getSources({ types: ['screen', 'window'] });
    const source = sources.find((item) => item.id === selectedSourceId) || sources[0];
    callback({ video: source, audio: 'loopback' });
  });

  ipcMain.handle('sources:list', async () => {
    const sources = await desktopCapturer.getSources({
      types: ['screen', 'window'],
      thumbnailSize: { width: 360, height: 210 },
      fetchWindowIcons: true
    });
    return sources.map(({ id, name, thumbnail, appIcon }) => ({
      id,
      name,
      thumbnail: thumbnail.toDataURL(),
      icon: appIcon ? appIcon.toDataURL() : null
    }));
  });

  ipcMain.handle('sources:select', (_event, id) => {
    selectedSourceId = id;
    return true;
  });

  ipcMain.handle('recording:save', async (_event, data) => {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const result = await dialog.showSaveDialog({
      title: 'Save recording',
      defaultPath: `Screen-Studio-${stamp}.webm`,
      filters: [{ name: 'WebM video', extensions: ['webm'] }]
    });
    if (result.canceled || !result.filePath) return null;
    await fs.writeFile(result.filePath, Buffer.from(data));
    return result.filePath;
  });

  createWindow();
  app.on('activate', () => BrowserWindow.getAllWindows().length === 0 && createWindow());
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
