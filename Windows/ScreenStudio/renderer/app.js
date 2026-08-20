const $ = (selector) => document.querySelector(selector);
const screenVideo = $('#screenVideo');
const cameraVideo = $('#cameraVideo');
const canvas = $('#previewCanvas');
const ctx = canvas.getContext('2d');
const state = { screenStream:null, cameraStream:null, micStream:null, recorder:null, recordingId:null, writeChain:Promise.resolve(), writeError:null, currentSource:null, drawing:false, startedAt:0, timer:null, region:{ x:.1,y:.1,w:.8,h:.8 } };

async function openSources() {
  const dialog = $('#sourceDialog');
  const grid = $('#sourceGrid');
  grid.innerHTML = '<p>Loading available screens…</p>';
  dialog.showModal();
  try {
    const sources = await window.screenStudio.listSources();
    grid.innerHTML = '';
    for (const source of sources) {
      const button = document.createElement('button');
      button.className = 'source-item';
      button.innerHTML = `<img src="${source.thumbnail}" alt=""><span>${escapeHtml(source.name)}</span>`;
      button.onclick = () => selectSource(source, dialog);
      grid.append(button);
    }
  } catch (error) { grid.innerHTML = `<p>Could not list screens: ${escapeHtml(error.message)}</p>`; }
}

async function selectSource(source, dialog) {
  await stopTracks(state.screenStream);
  await window.screenStudio.selectSource(source.id);
  try {
    state.screenStream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: Number($('#fps').value) }, audio: true });
    state.currentSource = source;
    screenVideo.srcObject = state.screenStream;
    await screenVideo.play();
    const track = state.screenStream.getVideoTracks()[0];
    const settings = track.getSettings();
    $('#sourceButton b').textContent = source.name;
    $('#sourceButton small').textContent = source.id.startsWith('screen') ? 'Entire display' : 'Application window';
    $('#resolutionLabel').textContent = `${settings.width} × ${settings.height} · ${settings.frameRate || $('#fps').value} FPS`;
    $('#emptyState').classList.add('hidden');
    canvas.style.display = 'block';
    $('#recordButton').disabled = false;
    $('#hint').textContent = 'Everything is ready';
    track.addEventListener('ended', resetSource, { once:true });
    beginPreview();
    updateRegionBox();
    dialog.close();
  } catch (error) { $('#hint').textContent = `Capture failed: ${error.message}`; }
}

function beginPreview() {
  if (state.drawing) return;
  state.drawing = true;
  const draw = () => {
    if (!state.drawing || !state.screenStream) return;
    const source = getCropPixels();
    if (canvas.width !== source.w || canvas.height !== source.h) { canvas.width = source.w; canvas.height = source.h; }
    ctx.drawImage(screenVideo, source.x, source.y, source.w, source.h, 0, 0, canvas.width, canvas.height);
    if (state.cameraStream && cameraVideo.readyState >= 2) drawCamera();
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

function getCropPixels() {
  const w = screenVideo.videoWidth || 1920, h = screenVideo.videoHeight || 1080;
  if (!$('#regionToggle').checked) return { x:0,y:0,w,h };
  return { x:Math.round(state.region.x*w), y:Math.round(state.region.y*h), w:Math.max(2,Math.round(state.region.w*w)), h:Math.max(2,Math.round(state.region.h*h)) };
}

function drawCamera() {
  const ratio = Number($('#cameraSize').value), diameter = canvas.width * ratio, margin = canvas.width * .025;
  const pos = $('#cameraPosition').value;
  const x = pos.endsWith('r') ? canvas.width-diameter/2-margin : diameter/2+margin;
  const y = pos.startsWith('b') ? canvas.height-diameter/2-margin : diameter/2+margin;
  ctx.save();ctx.beginPath();ctx.arc(x,y,diameter/2,0,Math.PI*2);ctx.clip();
  const scale = Math.max(diameter/cameraVideo.videoWidth,diameter/cameraVideo.videoHeight);
  const w=cameraVideo.videoWidth*scale,h=cameraVideo.videoHeight*scale;
  ctx.translate(x,y);ctx.scale(-1,1);ctx.drawImage(cameraVideo,-w/2,-h/2,w,h);ctx.restore();
  ctx.beginPath();ctx.arc(x,y,diameter/2,0,Math.PI*2);ctx.lineWidth=Math.max(4,canvas.width*.004);ctx.strokeStyle='#fff';ctx.stroke();
}

async function toggleCamera() {
  if ($('#cameraToggle').checked) {
    try { state.cameraStream = await navigator.mediaDevices.getUserMedia({ video:{ width:{ideal:1920},height:{ideal:1080} }, audio:false }); cameraVideo.srcObject=state.cameraStream; await cameraVideo.play(); }
    catch(error){ $('#cameraToggle').checked=false; $('#hint').textContent=`Camera unavailable: ${error.message}`; }
  } else { await stopTracks(state.cameraStream); state.cameraStream=null; }
}

async function startRecording() {
  if (!state.screenStream) return;
  $('#recordButton').disabled = true;
  try {
    const sourceVideoTrack = state.screenStream.getVideoTracks()[0];
    if (!sourceVideoTrack || sourceVideoTrack.readyState !== 'live') throw new Error('The selected screen is no longer producing video');
    const recordingId = await window.screenStudio.startRecordingFile();
    state.recordingId = recordingId;
    const audioContext = new AudioContext();
    await audioContext.resume();
    const destination = audioContext.createMediaStreamDestination();
    if ($('#systemToggle').checked && state.screenStream.getAudioTracks().length) audioContext.createMediaStreamSource(new MediaStream(state.screenStream.getAudioTracks())).connect(destination);
    if ($('#micToggle').checked) { state.micStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});audioContext.createMediaStreamSource(state.micStream).connect(destination); }
    const fps=Number($('#fps').value);
    // Record the native display track when no compositing is requested. This
    // avoids making ordinary full-screen recordings depend on canvas paints.
    // Region and camera recordings still use the compositor.
    const needsCompositor=$('#regionToggle').checked||$('#cameraToggle').checked;
    const output=needsCompositor?canvas.captureStream(fps):new MediaStream([sourceVideoTrack.clone()]);
    const outputVideoTrack=output.getVideoTracks()[0];
    if (!outputVideoTrack || outputVideoTrack.readyState !== 'live') throw new Error('Could not create a live video track for the recording');
    destination.stream.getAudioTracks().forEach(track=>output.addTrack(track));
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus') ? 'video/webm;codecs=vp9,opus' : 'video/webm;codecs=vp8,opus';
    state.writeChain=Promise.resolve();state.writeError=null;
    state.recorder=new MediaRecorder(output,{mimeType,videoBitsPerSecond:Number($('#quality').value),audioBitsPerSecond:192000});
    state.recorder.ondataavailable=e=>{
      if (!e.data.size) return;
      state.writeChain = state.writeChain.then(async()=>window.screenStudio.appendRecordingChunk(recordingId,await e.data.arrayBuffer())).catch(error=>{state.writeError=error;});
    };
    state.recorder.onerror=event=>{state.writeError=event.error||new Error('The video encoder stopped unexpectedly');};
    state.recorder.onstop=async()=>{
      $('#hint').textContent='Finalizing every recorded frame… please wait';
      try {
        await state.writeChain;
        if (state.writeError) throw state.writeError;
        const result=await window.screenStudio.finishRecordingFile(recordingId);
        $('#hint').textContent=result?`Saved ${(result.bytes/1073741824).toFixed(2)} GB to ${result.path}`:'Recording was not saved';
      } catch(error) {
        await window.screenStudio.abortRecordingFile(recordingId);
        $('#hint').textContent=`Could not save the complete recording: ${error.message}`;
      } finally {
        output.getTracks().forEach(track=>track.stop());
        await stopTracks(state.micStream);state.micStream=null;
        await audioContext.close();state.recordingId=null;state.recorder=null;setRecordingUi(false);
      }
    };
    state.recorder.start(1000);state.startedAt=Date.now();setRecordingUi(true);
  } catch(error) { if(state.recordingId)await window.screenStudio.abortRecordingFile(state.recordingId);state.recordingId=null;$('#hint').textContent=`Could not start: ${error.message}`;$('#recordButton').disabled=false; }
}

function stopRecording(){ if(state.recorder?.state==='recording'){state.recorder.requestData();state.recorder.stop();$('#recordButton').disabled=true;$('#hint').textContent='Stopping and flushing video data…';} }
function setRecordingUi(active){ const button=$('#recordButton');button.disabled=active?false:!state.screenStream;button.classList.toggle('stop',active);button.querySelector('span').textContent=active?'Stop & save':'Start recording';$('#statusPill').classList.toggle('recording',active);$('#statusPill span').textContent=active?'Recording':'Ready';clearInterval(state.timer);if(active){state.timer=setInterval(updateTimer,250);updateTimer();}else{$('#timer').textContent='00:00:00';} }
function updateTimer(){const s=Math.floor((Date.now()-state.startedAt)/1000);$('#timer').textContent=[Math.floor(s/3600),Math.floor(s%3600/60),s%60].map(v=>String(v).padStart(2,'0')).join(':');}
function resetSource(){if(state.recorder?.state==='recording')stopRecording();state.drawing=false;state.screenStream=null;state.currentSource=null;canvas.style.display='none';$('#emptyState').classList.remove('hidden');$('#recordButton').disabled=true;$('#resolutionLabel').textContent='No source selected';if(!state.recordingId)$('#hint').textContent='Select a source to start recording';}
async function stopTracks(stream){stream?.getTracks().forEach(track=>track.stop());}
function escapeHtml(value){const node=document.createElement('span');node.textContent=value;return node.innerHTML;}

function updateRegionBox(){const box=$('#regionBox');if(!$('#regionToggle').checked||!state.screenStream){box.classList.add('hidden');return;}box.classList.remove('hidden');box.style.left=`${state.region.x*100}%`;box.style.top=`${state.region.y*100}%`;box.style.width=`${state.region.w*100}%`;box.style.height=`${state.region.h*100}%`;}

async function toggleRegion(){
  const toggle=$('#regionToggle');
  if(!toggle.checked){updateRegionBox();return;}
  if(!state.currentSource){toggle.checked=false;$('#hint').textContent='Choose an entire display before selecting a region';return;}
  if(!state.currentSource.id.startsWith('screen')){toggle.checked=false;$('#hint').textContent='Region capture is available when an entire display is selected';return;}
  toggle.disabled=true;$('#hint').textContent='Select the recording area on your screen';
  try {
    const region=await window.screenStudio.selectRegion(state.currentSource.id);
    if(region){state.region=region;updateRegionBox();$('#hint').textContent='Region selected · ready to record';}
    else {toggle.checked=false;updateRegionBox();$('#hint').textContent='Region selection canceled';}
  } catch(error){toggle.checked=false;updateRegionBox();$('#hint').textContent=`Could not select region: ${error.message}`;}
  finally {toggle.disabled=false;}
}

$('#chooseSourceHero').onclick=openSources;$('#sourceButton').onclick=openSources;$('#refreshSources').onclick=openSources;$('#closeDialog').onclick=()=>$('#sourceDialog').close();$('#cameraToggle').onchange=toggleCamera;$('#regionToggle').onchange=toggleRegion;$('#recordButton').onclick=()=>state.recorder?.state==='recording'?stopRecording():startRecording();window.addEventListener('beforeunload',()=>{stopTracks(state.screenStream);stopTracks(state.cameraStream);stopTracks(state.micStream);});
