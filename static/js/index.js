// LoBE-GS project page scripts.
$(document).ready(function () {
  // Responsive navbar burger.
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Video gallery: a flat list of clips shown one per row.
  // The arrows page through them with a CSS slide transition. Because clips have
  // different aspect ratios, the viewport height follows the current clip so the
  // arrows re-center and the dots sit right under it.
  $(".video-gallery").each(function () {
    var $g = $(this);
    var videos = $g.data("videos"); // jQuery parses the JSON data-videos attribute.
    if (!videos || !videos.length) return;

    var $prev = $g.find(".vg-prev");
    var $next = $g.find(".vg-next");
    var $dots = $g.find(".vg-dots");
    var $viewport = $g.find(".vg-viewport");
    var $track = $g.find(".vg-track");
    var $caption = $g.find(".vg-caption");

    var per = 1;
    var pages = Math.ceil(videos.length / per);
    var page = 0;

    // Each entry may be a plain src string or an { src, label } object.
    function item(i) {
      var it = videos[i];
      return (typeof it === "string") ? { src: it, label: "" } : it;
    }

    // Build all clip items once.
    var html = "";
    for (var i = 0; i < videos.length; i++) {
      html += '<div class="vg-item"><div class="comparison-item">' +
                '<video controls muted loop playsinline preload="metadata" src="' + item(i).src + '"></video>' +
              '</div></div>';
    }
    $track.html(html);

    // Size the viewport to the current clip's rendered height.
    function fitHeight() {
      var v = $track.find("video").eq(page)[0];
      if (v && v.offsetHeight > 0) $viewport.css("height", v.offsetHeight + "px");
    }

    // Re-fit when a clip's dimensions become known, or on resize.
    $track.find("video").on("loadedmetadata loadeddata", function () {
      if ($track.find("video").index(this) === page) fitHeight();
    });
    $(window).on("resize", fitHeight);

    // Move to page `page`; animate unless `false` is passed.
    function position(animate) {
      if (page > pages - 1) page = pages - 1;
      if (page < 0) page = 0;

      if (!animate) $track.css("transition", "none");
      $track.css("transform", "translateX(-" + (page * 100) + "%)");
      if (!animate) {
        $track[0].offsetHeight;         // force reflow so the next transition applies
        $track.css("transition", "");
      }

      // Autoplay the visible clip (muted); pause the rest.
      $track.find("video").each(function (i) {
        if (i === page) {
          var pr = this.play();
          if (pr) pr.catch(function () { /* autoplay blocked; ignore */ });
        } else {
          this.pause();
        }
      });

      fitHeight();   // match the viewport to this clip

      // Caption naming the current clip's dataset.
      $caption.html(item(page).label || "");

      // Dots.
      $dots.empty();
      for (var k = 0; k < pages; k++) {
        (function (idx) {
          var $dot = $('<button type="button" class="ic2-dot" aria-label="Go to video"></button>');
          if (idx === page) $dot.addClass("is-active");
          $dot.on("click", function () { page = idx; position(true); });
          $dots.append($dot);
        })(k);
      }

      var multi = pages > 1;
      $prev.toggle(multi).prop("disabled", page === 0);
      $next.toggle(multi).prop("disabled", page === pages - 1);
      $dots.toggle(multi);
    }

    $prev.on("click", function () { if (page > 0) { page--; position(true); } });
    $next.on("click", function () { if (page < pages - 1) { page++; position(true); } });

    position(false);
  });

  // Image comparisons: a track of Ours-vs-baseline drag sliders (2 visible).
  // The arrows slide the track by one baseline with a CSS transition
  // (e.g. [CityGS, VastGS] -> [VastGS, Grendel]).
  $(".img-compare2").each(function () {
    var $c = $(this);
    var datasets = $c.data("datasets");
    if (!datasets || !datasets.length) return;

    var $tabs = $c.find(".vg-datasets");
    var $prev = $c.find(".ic-prev");
    var $next = $c.find(".ic-next");
    var $dots = $c.find(".ic2-dots");
    var $track = $c.find(".ic-track");

    var d = 0; // dataset index
    var p = 0; // window start index into ds.comparisons

    // Update one slider's wipe position.
    function setSplit(slider, pct) {
      pct = Math.max(2, Math.min(98, pct));
      slider.querySelector(".ic-topimg").style.clipPath = "inset(0 " + (100 - pct) + "% 0 0)";
      slider.querySelector(".ic-divider").style.left = pct + "%";
      slider.querySelector(".ic-handle").style.left = pct + "%";
    }

    // Shared drag state (delegated, so it works for dynamically built sliders).
    var dragging = null;
    function pctFromEvent(slider, e) {
      var rect = slider.getBoundingClientRect();
      return ((e.clientX - rect.left) / rect.width) * 100;
    }
    $track.on("pointerdown", ".ic-slider", function (e) {
      dragging = this;
      setSplit(this, pctFromEvent(this, e));
      e.preventDefault();
    });
    window.addEventListener("pointermove", function (e) {
      if (dragging) setSplit(dragging, pctFromEvent(dragging, e));
    });
    window.addEventListener("pointerup", function () { dragging = null; });

    // Each comparison carries its own ours/other image pair.
    function makeItem(cmp) {
      return '' +
        '<div class="ic-item"><div class="ic-slider">' +
          '<img class="ic-base" src="' + cmp.other + '" alt="' + (cmp.name || "Baseline") + '">' +
          '<img class="ic-topimg" src="' + cmp.ours + '" alt="Ours (LoBE-GS)">' +
          '<div class="ic-divider"></div>' +
          '<div class="ic-handle"><i class="fas fa-arrows-alt-h"></i></div>' +
          '<span class="ic-label ic-label-left">Ours</span>' +
          '<span class="ic-label ic-label-right">' + (cmp.name || "Baseline") + '</span>' +
        '</div>' +
        '<button class="ic-expand" type="button" aria-label="Enlarge"><i class="fas fa-expand"></i></button>' +
        '</div>';
    }

    // Rebuild the track for the active dataset (no slide animation on switch).
    function buildTrack() {
      var comparisons = datasets[d].comparisons || [];
      var html = "";
      for (var i = 0; i < comparisons.length; i++) html += makeItem(comparisons[i]);
      $track.html(html);
      $track.toggleClass("ic-single", comparisons.length <= 1);
      p = 0;
      position(false);
    }

    // Move the window to page p; animate unless `false` is passed.
    function position(animate) {
      var n = (datasets[d].comparisons || []).length;
      var pages = Math.max(1, n - 1);
      if (p > pages - 1) p = pages - 1;
      if (p < 0) p = 0;

      if (!animate) $track.css("transition", "none");
      $track.css("transform", "translateX(-" + (p * 50) + "%)");
      if (!animate) {
        $track[0].offsetHeight;         // force reflow so the next transition applies
        $track.css("transition", "");
      }

      // Dots.
      $dots.empty();
      for (var k = 0; k < pages; k++) {
        (function (idx) {
          var $dot = $('<button type="button" class="ic2-dot" aria-label="Go to comparison"></button>');
          if (idx === p) $dot.addClass("is-active");
          $dot.on("click", function () { p = idx; position(true); });
          $dots.append($dot);
        })(k);
      }

      var multi = pages > 1;
      $prev.toggle(multi).prop("disabled", p === 0);
      $next.toggle(multi).prop("disabled", p === pages - 1);
      $dots.toggle(multi);
    }

    // Dataset pill buttons.
    datasets.forEach(function (ds, idx) {
      var $b = $('<button type="button" class="button vg-ds-btn"></button>').text(ds.name);
      $b.on("click", function () {
        d = idx;
        $tabs.find(".vg-ds-btn").removeClass("is-link").eq(d).addClass("is-link");
        buildTrack();
      });
      $tabs.append($b);
    });

    $prev.on("click", function () { if (p > 0) { p--; position(true); } });
    $next.on("click", function () {
      var pages = Math.max(1, (datasets[d].comparisons || []).length - 1);
      if (p < pages - 1) { p++; position(true); }
    });

    // ---- Lightbox (enlarge on click) ----
    var $modal = $c.find(".ic-modal");
    var modalSlider = $c.find(".ic-modal-slider")[0];
    var $modalBase = $modal.find(".ic-base");
    var $modalTop = $modal.find(".ic-topimg");
    var $modalLabel = $modal.find(".ic-label-right");

    function openModal(oursSrc, otherSrc, name) {
      $modalBase.attr("src", otherSrc);
      $modalTop.attr("src", oursSrc);
      $modalLabel.text(name || "Baseline");
      setSplit(modalSlider, 50);
      $modal.addClass("is-open");
    }
    function closeModal() { $modal.removeClass("is-open"); }

    // Expand button on an item opens the lightbox for that comparison.
    $track.on("click", ".ic-expand", function () {
      var $sl = $(this).closest(".ic-item").find(".ic-slider");
      openModal($sl.find(".ic-topimg").attr("src"),
                $sl.find(".ic-base").attr("src"),
                $sl.find(".ic-label-right").text());
    });

    // Drag the enlarged slider too (reuses the shared drag state).
    modalSlider.addEventListener("pointerdown", function (e) {
      dragging = modalSlider;
      setSplit(modalSlider, pctFromEvent(modalSlider, e));
      e.preventDefault();
    });

    $c.find(".ic-modal-close").on("click", closeModal);
    $modal.on("click", function (e) { if (e.target === this) closeModal(); });   // click backdrop
    $(document).on("keydown", function (e) { if (e.key === "Escape") closeModal(); });

    // Initial state.
    $tabs.find(".vg-ds-btn").eq(0).addClass("is-link");
    buildTrack();
  });
});
