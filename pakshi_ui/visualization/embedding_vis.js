(function () {
  "use strict";

  const EFFNET_COLORS = {
    DBlue: 0x3a3d47,
    Grey: 0x8b8e94,
    LBlue: 0xc4c0ba,
    M: 0xd94030,
    Orange: 0xe8963c,
    Pink: 0xb07858,
    White: 0xf0ede6,
    Yellow: 0xd4b896,
    Unknown: 0xb8b0a4,
  };

  const CREPE_COLORS = {
    DBlue: 0xc5c2b8,
    Grey: 0x74716b,
    LBlue: 0x3b3f45,
    M: 0x26bfcf,
    Orange: 0x1769c3,
    Pink: 0x4f87a7,
    White: 0x0f1219,
    Yellow: 0x2b4769,
    Unknown: 0x8ea3b8,
  };

  const LIVE_COLOR = 0xff5959;
  const HIGHLIGHT_COLOR = 0x00ff88;
  const DEFAULT_POINT_SCALE = 0.04;
  const HIGHLIGHT_DURATION = 500;

  let palette = EFFNET_COLORS;
  let defaultColor = 0x8b8e94;
  let scene;
  let camera;
  let renderer;
  let controls;
  let corpusGroup;
  let liveGroup;
  let linesGroup;
  let corpusPoints = [];
  let highlightQueue = [];
  let activeHighlight = null;
  let pendingCorpus = null;
  let pendingStatus = null;
  let focusTarget = null;
  let gaussianTex = null;

  function makeGaussianTexture(size) {
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    const c = size / 2;
    const grad = ctx.createRadialGradient(c, c, 0, c, c, c);
    grad.addColorStop(0, "rgba(255,255,255,1.0)");
    grad.addColorStop(0.15, "rgba(255,255,255,0.6)");
    grad.addColorStop(0.45, "rgba(255,255,255,0.14)");
    grad.addColorStop(1, "rgba(255,255,255,0.0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    return tex;
  }

  function getGaussianTex() {
    if (!gaussianTex) {
      gaussianTex = makeGaussianTexture(128);
    }
    return gaussianTex;
  }

  function applyPalette(modelPath) {
    const useCrepe = /crepe/i.test(String(modelPath || ""));
    palette = useCrepe ? CREPE_COLORS : EFFNET_COLORS;
    defaultColor = useCrepe ? 0x74716b : 0x8b8e94;
    document.title = useCrepe
      ? "Pakshi Machina — CREPE Embedding Space"
      : "Pakshi Machina — EffNet-Bio Embedding Space";
    document.body.dataset.family = useCrepe ? "crepe" : "effnet";
    if (corpusPoints.length) {
      for (const point of corpusPoints) {
        point.baseColor = palette[point.syl_type] || defaultColor;
        restorePoint(point);
      }
    }
  }

  function init(container) {
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(35, width / height, 0.01, 100);
    camera.position.set(1.8, 1.2, 1.8);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x0b0d11, 1);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.35;
    controls.minDistance = 0.5;
    controls.maxDistance = 8;

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    corpusGroup = new THREE.Group();
    liveGroup = new THREE.Group();
    linesGroup = new THREE.Group();
    scene.add(corpusGroup);
    scene.add(liveGroup);
    scene.add(linesGroup);

    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(container);

    animate();
    if (pendingCorpus) {
      setCorpus(pendingCorpus);
      pendingCorpus = null;
    }
    if (pendingStatus) {
      setStatus(pendingStatus);
      pendingStatus = null;
    }
  }

  function setCorpus(points) {
    if (!scene) {
      pendingCorpus = points;
      return;
    }
    while (corpusGroup.children.length) {
      corpusGroup.remove(corpusGroup.children[0]);
    }
    corpusPoints = [];
    const tex = getGaussianTex();
    for (const pt of points || []) {
      const color = palette[pt.syl_type] || defaultColor;
      const mat = new THREE.SpriteMaterial({
        map: tex,
        color,
        transparent: true,
        opacity: 0.55,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const sprite = new THREE.Sprite(mat);
      sprite.position.set(pt.x, pt.y, pt.z);
      sprite.scale.set(DEFAULT_POINT_SCALE, DEFAULT_POINT_SCALE, 1);
      corpusGroup.add(sprite);
      corpusPoints.push({
        mesh: sprite,
        index: pt.index,
        label: pt.label,
        syl_type: pt.syl_type,
        baseColor: color,
      });
    }
  }

  function restorePoint(point) {
    point.mesh.material.color.setHex(point.baseColor);
    point.mesh.material.opacity = 0.55;
    point.mesh.scale.set(DEFAULT_POINT_SCALE, DEFAULT_POINT_SCALE, 1);
  }

  function resetAll() {
    highlightQueue = [];
    if (activeHighlight) {
      restorePoint(activeHighlight.point);
      activeHighlight = null;
    }
    for (const point of corpusPoints) {
      restorePoint(point);
    }
    while (liveGroup.children.length) {
      liveGroup.remove(liveGroup.children[0]);
    }
    while (linesGroup.children.length) {
      linesGroup.remove(linesGroup.children[0]);
    }
  }

  function highlightMatches(matches) {
    const now = performance.now();
    highlightQueue = [];
    for (const match of matches || []) {
      highlightQueue.push({
        index: match.corpus_index,
        startTime: now + Number(match.scheduled_offset_seconds || 0) * 1000,
      });
    }
  }

  function updateHighlights() {
    const now = performance.now();
    if (activeHighlight && now - activeHighlight.startTime >= HIGHLIGHT_DURATION) {
      restorePoint(activeHighlight.point);
      activeHighlight = null;
    }
    if (activeHighlight) {
      const t = (now - activeHighlight.startTime) / HIGHLIGHT_DURATION;
      const scale = 1 + 0.45 * Math.sin(t * Math.PI);
      activeHighlight.point.mesh.scale.set(DEFAULT_POINT_SCALE * scale, DEFAULT_POINT_SCALE * scale, 1);
    }
    if (!activeHighlight && highlightQueue.length && now >= highlightQueue[0].startTime) {
      const next = highlightQueue.shift();
      const point = corpusPoints.find((item) => item.index === next.index);
      if (!point) {
        return;
      }
      point.mesh.material.color.setHex(HIGHLIGHT_COLOR);
      point.mesh.material.opacity = 0.9;
      activeHighlight = { point, startTime: now };
    }
  }

  function addLivePoint(x, y, z) {
    const mat = new THREE.SpriteMaterial({
      map: getGaussianTex(),
      color: LIVE_COLOR,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(x, y, z);
    sprite.scale.set(DEFAULT_POINT_SCALE * 1.5, DEFAULT_POINT_SCALE * 1.5, 1);
    sprite.userData.born = performance.now();
    liveGroup.add(sprite);
  }

  function updateLivePoints() {
    const now = performance.now();
    const toRemove = [];
    for (const child of liveGroup.children) {
      const age = now - child.userData.born;
      if (age > 8000) {
        toRemove.push(child);
      } else {
        child.material.opacity = 1 - age / 8000;
      }
    }
    for (const child of toRemove) {
      liveGroup.remove(child);
    }
  }

  function addConnectionLines(queryPositions, matches) {
    while (linesGroup.children.length) {
      linesGroup.remove(linesGroup.children[0]);
    }
    for (let i = 0; i < Math.min(queryPositions.length, matches.length); i += 1) {
      const qp = queryPositions[i];
      const mp = matches[i].pos3d;
      if (!qp || !mp) {
        continue;
      }
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(qp.x, qp.y, qp.z),
        new THREE.Vector3(mp.x, mp.y, mp.z),
      ]);
      const mat = new THREE.LineBasicMaterial({ color: LIVE_COLOR, transparent: true, opacity: 0.4 });
      const line = new THREE.Line(geo, mat);
      line.userData.born = performance.now();
      linesGroup.add(line);
    }
  }

  function updateLines() {
    const now = performance.now();
    const toRemove = [];
    for (const child of linesGroup.children) {
      const age = now - child.userData.born;
      if (age > 8000) {
        toRemove.push(child);
      } else {
        child.material.opacity = 0.4 * (1 - age / 8000);
      }
    }
    for (const child of toRemove) {
      linesGroup.remove(child);
    }
  }

  function focusOnPoints(positions) {
    if (!positions || !positions.length) {
      return;
    }
    const cx = positions.reduce((sum, p) => sum + p.x, 0) / positions.length;
    const cy = positions.reduce((sum, p) => sum + p.y, 0) / positions.length;
    const cz = positions.reduce((sum, p) => sum + p.z, 0) / positions.length;
    focusTarget = { x: cx, y: cy, z: cz, startTime: performance.now() };
  }

  function updateFocus() {
    if (!focusTarget) {
      return;
    }
    const t = Math.min(1, (performance.now() - focusTarget.startTime) / 1500);
    const ease = t * (2 - t);
    controls.target.x += (focusTarget.x - controls.target.x) * ease * 0.08;
    controls.target.y += (focusTarget.y - controls.target.y) * ease * 0.08;
    controls.target.z += (focusTarget.z - controls.target.z) * ease * 0.08;
    if (t >= 1) {
      focusTarget = null;
    }
  }

  function addLivePoints(queryPositions, matches) {
    for (const point of queryPositions || []) {
      addLivePoint(point.x, point.y, point.z);
    }
    if (queryPositions && matches) {
      addConnectionLines(queryPositions, matches);
      focusOnPoints(queryPositions);
    }
  }

  function setStatus(text) {
    pendingStatus = text;
    const el = document.getElementById("status");
    if (el) {
      el.textContent = text;
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    if (!renderer) {
      return;
    }
    controls.update();
    updateHighlights();
    updateLivePoints();
    updateLines();
    updateFocus();
    renderer.render(scene, camera);
  }

  window.embeddingVis = {
    init,
    setCorpus,
    resetAll,
    applyPalette,
    highlightMatches,
    addLivePoints,
    setStatus,
  };
})();
