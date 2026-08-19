# Screen Studio for Windows

A high-quality desktop screen recorder built with Electron. It can record a full display or application window, crop to a movable/resizable region, capture microphone and Windows system audio, and add a circular webcam overlay.

## Run locally

Install [Node.js 20+](https://nodejs.org/) on Windows, then open PowerShell in this folder:

```powershell
npm install
npm start
```

## Build a Windows installer

```powershell
npm run dist
```

The installer and portable executable are created in `dist/`. Build the distributable on Windows so Electron Builder can create the Windows packages reliably.

## Recording format

Recordings use VP9/Opus WebM at 6, 12, or 24 Mbps and up to 60 FPS. WebM is used because Chromium can encode it reliably in real time without native codecs. Files play in modern browsers, VLC, and current Windows media apps; they can be converted to MP4 afterward with FFmpeg if required.

## Notes

- System audio loopback is provided by Electron on Windows.
- The final recording resolution matches the selected source or selected crop region.
- Microphone processing enables echo cancellation, noise suppression, and automatic gain control.
- A stopped screen share automatically stops and offers to save an active recording.
