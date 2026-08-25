/*
 * Click-to-zoom for images in post content.
 *
 * Any <img> inside .post-content becomes clickable and opens in an overlay.
 * Inside the overlay, clicking the image toggles between fit-to-screen and
 * actual size (with panning) — useful for wide diagrams.
 *
 * Opt an image out by wrapping it in a link, or adding class="no-zoom".
 * Close with Escape, a click on the backdrop, or the close button.
 */
(function () {
  "use strict";

  var SELECTOR = ".post-content img:not(.no-zoom)";
  var overlay, figure, img, caption, closeBtn, lastFocused, zoomed;

  function build() {
    overlay = document.createElement("div");
    overlay.className = "lightbox";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Image viewer");
    overlay.hidden = true;

    closeBtn = document.createElement("button");
    closeBtn.className = "lightbox__close";
    closeBtn.type = "button";
    closeBtn.setAttribute("aria-label", "Close image viewer");
    closeBtn.innerHTML = "&times;";

    figure = document.createElement("figure");
    figure.className = "lightbox__figure";

    img = document.createElement("img");
    img.className = "lightbox__img";
    img.alt = "";

    caption = document.createElement("figcaption");
    caption.className = "lightbox__caption";

    figure.appendChild(img);
    figure.appendChild(caption);
    overlay.appendChild(closeBtn);
    overlay.appendChild(figure);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", function (e) {
      // Clicks on the image toggle zoom; anywhere else closes.
      if (e.target === img) {
        toggleZoom(e);
      } else if (e.target !== caption) {
        close();
      }
    });
    closeBtn.addEventListener("click", close);
    document.addEventListener("keydown", function (e) {
      if (!overlay.hidden && (e.key === "Escape" || e.key === "Esc")) close();
    });
  }

  function open(source) {
    if (!overlay) build();
    lastFocused = document.activeElement;

    img.src = source.currentSrc || source.src;
    img.alt = source.alt || "";
    var text = source.alt || "";
    caption.textContent = text;
    caption.hidden = !text;

    setZoom(false);
    overlay.hidden = false;
    document.body.classList.add("lightbox-open");
    closeBtn.focus();
  }

  function close() {
    if (!overlay || overlay.hidden) return;
    overlay.hidden = true;
    document.body.classList.remove("lightbox-open");
    setZoom(false);
    img.removeAttribute("src");
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  function setZoom(on) {
    zoomed = on;
    overlay.classList.toggle("is-zoomed", on);
  }

  function toggleZoom(e) {
    // Remember where the click landed as a fraction of the image, so the same
    // point stays under the cursor after switching to actual size.
    var rect = img.getBoundingClientRect();
    var fx = (e.clientX - rect.left) / rect.width;
    var fy = (e.clientY - rect.top) / rect.height;

    setZoom(!zoomed);

    if (zoomed) {
      overlay.scrollLeft = fx * overlay.scrollWidth - overlay.clientWidth / 2;
      overlay.scrollTop = fy * overlay.scrollHeight - overlay.clientHeight / 2;
    }
  }

  function init() {
    var images = document.querySelectorAll(SELECTOR);
    Array.prototype.forEach.call(images, function (el) {
      if (el.closest("a")) return; // already links somewhere — leave it alone
      el.classList.add("zoomable");
      el.addEventListener("click", function () {
        open(el);
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
