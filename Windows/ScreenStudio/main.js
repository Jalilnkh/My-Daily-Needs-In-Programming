const { app, BrowserWindow, desktopCapturer, dialog, ipcMain, session } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

let selectedSourceId = null;
const recordings = new Map();
let regionWindow = null;

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
      nodeIntegration: false,
      // The canvas compositor supplies recorded frames. It must keep rendering
      // while this window is covered or minimized during a screen recording.
      backgroundThrottling: false
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
    return sources.map(({ id, name, thumbnail, appIcon, display_id }) => ({
      id,
      name,
      displayId: display_id,
      thumbnail: thumbnail.toDataURL(),
      icon: appIcon ? appIcon.toDataURL() : null
    }));
  });

  ipcMain.handle('sources:select', (_event, id) => {
    selectedSourceId = id;
    return true;
  });

  ipcMain.handle('recording:start', async () => {
    const id = crypto.randomUUID();
    const tempPath = path.join(app.getPath('temp'), `screen-studio-${id}.webm`);
    const handle = await fs.open(tempPath, 'wx');
    recordings.set(id, { handle, tempPath, writeChain: Promise.resolve(), bytes: 0 });
    return id;
  });

  ipcMain.handle('recording:append', async (_event, id, data) => {
    const recording = recordings.get(id);
    if (!recording) throw new Error('Recording session is no longer available');
    const chunk = Buffer.from(data);
    recording.writeChain = recording.writeChain.then(async () => {
      await recording.handle.write(chunk);
      recording.bytes += chunk.length;
    });
    await recording.writeChain;
    return recording.bytes;
  });

  ipcMain.handle('recording:finish', async (_event, id) => {
    const recording = recordings.get(id);
    if (!recording) throw new Error('Recording session is no longer available');
    recordings.delete(id);
    await recording.writeChain;
    await recording.handle.sync();
    await recording.handle.close();
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const result = await dialog.showSaveDialog({
      title: 'Save recording',
      defaultPath: `Screen-Studio-${stamp}.webm`,
      filters: [{ name: 'WebM video', extensions: ['webm'] }]
    });
    if (result.canceled || !result.filePath) {
      await fs.unlink(recording.tempPath).catch(() => {});
      return null;
    }
    try {
      await fs.rename(recording.tempPath, result.filePath);
    } catch (error) {
      if (error.code !== 'EXDEV') throw error;
      await fs.copyFile(recording.tempPath, result.filePath);
      await fs.unlink(recording.tempPath);
    }
    return { path: result.filePath, bytes: recording.bytes };
  });

  ipcMain.handle('recording:abort', async (_event, id) => {
    const recording = recordings.get(id);
    if (!recording) return;
    recordings.delete(id);
    await recording.writeChain.catch(() => {});
    await recording.handle.close().catch(() => {});
    await fs.unlink(recording.tempPath).catch(() => {});
  });

  ipcMain.handle('region:select', async (_event, sourceId) => {
    if (regionWindow) return null;
    const sources = await desktopCapturer.getSources({ types: ['screen'] });
    const source = sources.find(item => item.id === sourceId);
    if (!source) throw new Error('Region selection is available for entire displays only');
    const display = require('electron').screen.getAllDisplays().find(item => String(item.id) === String(source.display_id))
      || require('electron').screen.getDisplayNearestPoint(require('electron').screen.getCursorScreenPoint());
    return new Promise(resolve => {
      let settled = false;
      const finish = value => {
        if (settled) return;
        settled = true;
        ipcMain.removeListener('region:selected', selected);
        if (regionWindow && !regionWindow.isDestroyed()) regionWindow.close();
        regionWindow = null;
        resolve(value);
      };
      const selected = (_selectedEvent, value) => finish(value);
      ipcMain.once('region:selected', selected);
      regionWindow = new BrowserWindow({
        ...display.bounds,
        frame: false,
        transparent: true,
        resizable: false,
        movable: false,
        alwaysOnTop: true,
        skipTaskbar: true,
        fullscreenable: false,
        webPreferences: {
          preload: path.join(__dirname, 'region-preload.js'),
          contextIsolation: true,
          nodeIntegration: false
        }
      });
      regionWindow.setAlwaysOnTop(true, 'screen-saver');
      regionWindow.loadFile(path.join(__dirname, 'renderer', 'region.html'));
      regionWindow.on('closed', () => finish(null));
    });
  });

  createWindow();
  app.on('activate', () => BrowserWindow.getAllWindows().length === 0 && createWindow());
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
