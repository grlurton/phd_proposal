! function(a, b) {
    "use strict";
    "function" == typeof define && define.amd ? define(function() {
        return a.Reveal = b(), a.Reveal
    }) : "object" == typeof exports ? module.exports = b() : a.Reveal = b()
}(this, function() {
    "use strict";

    function a(a) {
        if (b(), Hc.transforms2d || Hc.transforms3d) {
            Gc.wrapper = document.querySelector(".reveal"), Gc.slides = document.querySelector(".reveal .slides"), window.addEventListener("load", L, !1);
            var d = oc.getQueryHash();
            "undefined" != typeof d.dependencies && delete d.dependencies, p(Ac, a), p(Ac, d), B(), c()
        } else {
            document.body.setAttribute("class", "no-transforms");
            for (var e = document.getElementsByTagName("img"), f = 0, g = e.length; g > f; f++) {
                var h = e[f];
                h.getAttribute("data-src") && (h.setAttribute("src", h.getAttribute("data-src")), h.removeAttribute("data-src"))
            }
        }
    }

    function b() {
        Hc.transforms3d = "WebkitPerspective" in document.body.style || "MozPerspective" in document.body.style || "msPerspective" in document.body.style || "OPerspective" in document.body.style || "perspective" in document.body.style, Hc.transforms2d = "WebkitTransform" in document.body.style || "MozTransform" in document.body.style || "msTransform" in document.body.style || "OTransform" in document.body.style || "transform" in document.body.style, Hc.requestAnimationFrameMethod = window.requestAnimationFrame || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame, Hc.requestAnimationFrame = "function" == typeof Hc.requestAnimationFrameMethod, Hc.canvas = !!document.createElement("canvas").getContext, Hc.touch = !!("ontouchstart" in window), Hc.overviewTransitions = !/Version\/[\d\.]+.*Safari/.test(navigator.userAgent), uc = /(iphone|ipod|ipad|android)/gi.test(navigator.userAgent)
    }

    function c() {
        function a() {
            e.length && head.js.apply(null, e), d()
        }

        function b(b) {
            head.ready(b.src.match(/([\w\d_\-]*)\.?js$|[^\\\/]*$/i)[0], function() {
                "function" == typeof b.callback && b.callback.apply(this), 0 === --f && a()
            })
        }
        for (var c = [], e = [], f = 0, g = 0, h = Ac.dependencies.length; h > g; g++) {
            var i = Ac.dependencies[g];
            (!i.condition || i.condition()) && (i.async ? e.push(i.src) : c.push(i.src), b(i))
        }
        c.length ? (f = c.length, head.js.apply(null, c)) : a()
    }

    function d() {
        e(), l(), h(), eb(), m(), wb(), lb(!0), setTimeout(function() {
            Gc.slides.classList.remove("no-transition"), Bc = !0, D("ready", {
                indexh: pc,
                indexv: qc,
                currentSlide: sc
            })
        }, 1), A() && (o(), "complete" === document.readyState ? g() : window.addEventListener("load", g))
    }

    function e() {
        Gc.slides.classList.add("no-transition"), Gc.background = i(Gc.wrapper, "div", "backgrounds", null), Gc.progress = i(Gc.wrapper, "div", "progress", "<span></span>"), Gc.progressbar = Gc.progress.querySelector("span"), i(Gc.wrapper, "aside", "controls", '<div class="navigate-left"></div><div class="navigate-right"></div><div class="navigate-up"></div><div class="navigate-down"></div>'), Gc.slideNumber = i(Gc.wrapper, "div", "slide-number", ""), i(Gc.wrapper, "div", "pause-overlay", null), Gc.controls = document.querySelector(".reveal .controls"), Gc.theme = document.querySelector("#theme"), Gc.wrapper.setAttribute("role", "application"), Gc.controlsLeft = q(document.querySelectorAll(".navigate-left")), Gc.controlsRight = q(document.querySelectorAll(".navigate-right")), Gc.controlsUp = q(document.querySelectorAll(".navigate-up")), Gc.controlsDown = q(document.querySelectorAll(".navigate-down")), Gc.controlsPrev = q(document.querySelectorAll(".navigate-prev")), Gc.controlsNext = q(document.querySelectorAll(".navigate-next")), Gc.statusDiv = f()
    }

    function f() {
        var a = document.getElementById("aria-status-div");
        return a || (a = document.createElement("div"), a.style.position = "absolute", a.style.height = "1px", a.style.width = "1px", a.style.overflow = "hidden", a.style.clip = "rect( 1px, 1px, 1px, 1px )", a.setAttribute("id", "aria-status-div"), a.setAttribute("aria-live", "polite"), a.setAttribute("aria-atomic", "true"), Gc.wrapper.appendChild(a)), a
    }

    function g() {
        var a = N(window.innerWidth, window.innerHeight),
            b = Math.floor(a.width * (1 + Ac.margin)),
            c = Math.floor(a.height * (1 + Ac.margin)),
            d = a.width,
            e = a.height;
        v("@page{size:" + b + "px " + c + "px; margin: 0;}"), v(".reveal section>img, .reveal section>video, .reveal section>iframe{max-width: " + d + "px; max-height:" + e + "px}"), document.body.classList.add("print-pdf"), document.body.style.width = b + "px", document.body.style.height = c + "px", q(Gc.wrapper.querySelectorAll(wc)).forEach(function(a) {
            if (a.classList.contains("stack") === !1) {
                var f = (b - d) / 2,
                    g = (c - e) / 2,
                    h = y(a),
                    i = Math.max(Math.ceil(h / c), 1);
                (1 === i && Ac.center || a.classList.contains("center")) && (g = Math.max((c - h) / 2, 0)), a.style.left = f + "px", a.style.top = g + "px", a.style.width = d + "px";
                var j = a.querySelector(".slide-background");
                j && (j.style.width = b + "px", j.style.height = c * i + "px", j.style.top = -g + "px", j.style.left = -f + "px")
            }
        }), q(Gc.wrapper.querySelectorAll(wc + " .fragment")).forEach(function(a) {
            a.classList.add("visible")
        })
    }

    function h() {
        Gc.slides.querySelector("iframe") && setInterval(function() {
            (0 !== Gc.wrapper.scrollTop || 0 !== Gc.wrapper.scrollLeft) && (Gc.wrapper.scrollTop = 0, Gc.wrapper.scrollLeft = 0)
        }, 500)
    }

    function i(a, b, c, d) {
        for (var e = a.querySelectorAll("." + c), f = 0; f < e.length; f++) {
            var g = e[f];
            if (g.parentNode === a) return g
        }
        var h = document.createElement(b);
        return h.classList.add(c), "string" == typeof d && (h.innerHTML = d), a.appendChild(h), h
    }

    function j() {
        var a = A();
        Gc.background.innerHTML = "", Gc.background.classList.add("no-transition"), q(Gc.wrapper.querySelectorAll(xc)).forEach(function(b) {
            var c;
            c = a ? k(b, b) : k(b, Gc.background), q(b.querySelectorAll("section")).forEach(function(b) {
                a ? k(b, b) : k(b, c), c.classList.add("stack")
            })
        }), Ac.parallaxBackgroundImage ? (Gc.background.style.backgroundImage = 'url("' + Ac.parallaxBackgroundImage + '")', Gc.background.style.backgroundSize = Ac.parallaxBackgroundSize, setTimeout(function() {
            Gc.wrapper.classList.add("has-parallax-background")
        }, 1)) : (Gc.background.style.backgroundImage = "", Gc.wrapper.classList.remove("has-parallax-background"))
    }

    function k(a, b) {
        var c = {
                background: a.getAttribute("data-background"),
                backgroundSize: a.getAttribute("data-background-size"),
                backgroundImage: a.getAttribute("data-background-image"),
                backgroundVideo: a.getAttribute("data-background-video"),
                backgroundIframe: a.getAttribute("data-background-iframe"),
                backgroundColor: a.getAttribute("data-background-color"),
                backgroundRepeat: a.getAttribute("data-background-repeat"),
                backgroundPosition: a.getAttribute("data-background-position"),
                backgroundTransition: a.getAttribute("data-background-transition")
            },
            d = document.createElement("div");
        d.className = "slide-background " + a.className.replace(/present|past|future/, ""), c.background && (/^(http|file|\/\/)/gi.test(c.background) || /\.(svg|png|jpg|jpeg|gif|bmp)$/gi.test(c.background) ? a.setAttribute("data-background-image", c.background) : d.style.background = c.background), (c.background || c.backgroundColor || c.backgroundImage || c.backgroundVideo || c.backgroundIframe) && d.setAttribute("data-background-hash", c.background + c.backgroundSize + c.backgroundImage + c.backgroundVideo + c.backgroundIframe + c.backgroundColor + c.backgroundRepeat + c.backgroundPosition + c.backgroundTransition), c.backgroundSize && (d.style.backgroundSize = c.backgroundSize), c.backgroundColor && (d.style.backgroundColor = c.backgroundColor), c.backgroundRepeat && (d.style.backgroundRepeat = c.backgroundRepeat), c.backgroundPosition && (d.style.backgroundPosition = c.backgroundPosition), c.backgroundTransition && d.setAttribute("data-background-transition", c.backgroundTransition), b.appendChild(d), a.classList.remove("has-dark-background"), a.classList.remove("has-light-background");
        var e = window.getComputedStyle(d).backgroundColor;
        if (e) {
            var f = w(e);
            f && 0 !== f.a && a.classList.add(x(e) < 128 ? "has-dark-background" : "has-light-background")
        }
        return d
    }

    function l() {
        Ac.postMessage && window.addEventListener("message", function(a) {
            var b = a.data;
            "string" == typeof b && "{" === b.charAt(0) && "}" === b.charAt(b.length - 1) && (b = JSON.parse(b), b.method && "function" == typeof oc[b.method] && oc[b.method].apply(oc, b.args))
        }, !1)
    }

    function m(a) {
        var b = Gc.wrapper.querySelectorAll(wc).length;
        Gc.wrapper.classList.remove(Ac.transition), "object" == typeof a && p(Ac, a), Hc.transforms3d === !1 && (Ac.transition = "linear"), Gc.wrapper.classList.add(Ac.transition), Gc.wrapper.setAttribute("data-transition-speed", Ac.transitionSpeed), Gc.wrapper.setAttribute("data-background-transition", Ac.backgroundTransition), Gc.controls.style.display = Ac.controls ? "block" : "none", Gc.progress.style.display = Ac.progress ? "block" : "none", Ac.rtl ? Gc.wrapper.classList.add("rtl") : Gc.wrapper.classList.remove("rtl"), Ac.center ? Gc.wrapper.classList.add("center") : Gc.wrapper.classList.remove("center"), Ac.pause === !1 && Z(), Ac.mouseWheel ? (document.addEventListener("DOMMouseScroll", _b, !1), document.addEventListener("mousewheel", _b, !1)) : (document.removeEventListener("DOMMouseScroll", _b, !1), document.removeEventListener("mousewheel", _b, !1)), Ac.rollingLinks ? E() : F(), Ac.previewLinks ? G() : (H(), G("[data-preview-link]")), vc && (vc.destroy(), vc = null), b > 1 && Ac.autoSlide && Ac.autoSlideStoppable && Hc.canvas && Hc.requestAnimationFrame && (vc = new nc(Gc.wrapper, function() {
            return Math.min(Math.max((Date.now() - Nc) / Lc, 0), 1)
        }), vc.on("click", mc), Oc = !1), Ac.fragments === !1 && q(Gc.slides.querySelectorAll(".fragment")).forEach(function(a) {
            a.classList.add("visible"), a.classList.remove("current-fragment")
        }), db()
    }

    function n() {
        if (Kc = !0, window.addEventListener("hashchange", hc, !1), window.addEventListener("resize", ic, !1), Ac.touch && (Gc.wrapper.addEventListener("touchstart", Vb, !1), Gc.wrapper.addEventListener("touchmove", Wb, !1), Gc.wrapper.addEventListener("touchend", Xb, !1), window.navigator.pointerEnabled ? (Gc.wrapper.addEventListener("pointerdown", Yb, !1), Gc.wrapper.addEventListener("pointermove", Zb, !1), Gc.wrapper.addEventListener("pointerup", $b, !1)) : window.navigator.msPointerEnabled && (Gc.wrapper.addEventListener("MSPointerDown", Yb, !1), Gc.wrapper.addEventListener("MSPointerMove", Zb, !1), Gc.wrapper.addEventListener("MSPointerUp", $b, !1))), Ac.keyboard && (document.addEventListener("keydown", Ub, !1), document.addEventListener("keypress", Tb, !1)), Ac.progress && Gc.progress && Gc.progress.addEventListener("click", ac, !1), Ac.focusBodyOnPageVisibilityChange) {
            var a;
            "hidden" in document ? a = "visibilitychange" : "msHidden" in document ? a = "msvisibilitychange" : "webkitHidden" in document && (a = "webkitvisibilitychange"), a && document.addEventListener(a, jc, !1)
        }
        var b = ["touchstart", "click"];
        navigator.userAgent.match(/android/gi) && (b = ["touchstart"]), b.forEach(function(a) {
            Gc.controlsLeft.forEach(function(b) {
                b.addEventListener(a, bc, !1)
            }), Gc.controlsRight.forEach(function(b) {
                b.addEventListener(a, cc, !1)
            }), Gc.controlsUp.forEach(function(b) {
                b.addEventListener(a, dc, !1)
            }), Gc.controlsDown.forEach(function(b) {
                b.addEventListener(a, ec, !1)
            }), Gc.controlsPrev.forEach(function(b) {
                b.addEventListener(a, fc, !1)
            }), Gc.controlsNext.forEach(function(b) {
                b.addEventListener(a, gc, !1)
            })
        })
    }

    function o() {
        Kc = !1, document.removeEventListener("keydown", Ub, !1), document.removeEventListener("keypress", Tb, !1), window.removeEventListener("hashchange", hc, !1), window.removeEventListener("resize", ic, !1), Gc.wrapper.removeEventListener("touchstart", Vb, !1), Gc.wrapper.removeEventListener("touchmove", Wb, !1), Gc.wrapper.removeEventListener("touchend", Xb, !1), window.navigator.pointerEnabled ? (Gc.wrapper.removeEventListener("pointerdown", Yb, !1), Gc.wrapper.removeEventListener("pointermove", Zb, !1), Gc.wrapper.removeEventListener("pointerup", $b, !1)) : window.navigator.msPointerEnabled && (Gc.wrapper.removeEventListener("MSPointerDown", Yb, !1), Gc.wrapper.removeEventListener("MSPointerMove", Zb, !1), Gc.wrapper.removeEventListener("MSPointerUp", $b, !1)), Ac.progress && Gc.progress && Gc.progress.removeEventListener("click", ac, !1), ["touchstart", "click"].forEach(function(a) {
            Gc.controlsLeft.forEach(function(b) {
                b.removeEventListener(a, bc, !1)
            }), Gc.controlsRight.forEach(function(b) {
                b.removeEventListener(a, cc, !1)
            }), Gc.controlsUp.forEach(function(b) {
                b.removeEventListener(a, dc, !1)
            }), Gc.controlsDown.forEach(function(b) {
                b.removeEventListener(a, ec, !1)
            }), Gc.controlsPrev.forEach(function(b) {
                b.removeEventListener(a, fc, !1)
            }), Gc.controlsNext.forEach(function(b) {
                b.removeEventListener(a, gc, !1)
            })
        })
    }

    function p(a, b) {
        for (var c in b) a[c] = b[c]
    }

    function q(a) {
        return Array.prototype.slice.call(a)
    }

    function r(a) {
        if ("string" == typeof a) {
            if ("null" === a) return null;
            if ("true" === a) return !0;
            if ("false" === a) return !1;
            if (a.match(/^\d+$/)) return parseFloat(a)
        }
        return a
    }

    function s(a, b) {
        var c = a.x - b.x,
            d = a.y - b.y;
        return Math.sqrt(c * c + d * d)
    }

    function t(a, b) {
        a.style.WebkitTransform = b, a.style.MozTransform = b, a.style.msTransform = b, a.style.transform = b
    }

    function u(a) {
        "string" == typeof a.layout && (Fc.layout = a.layout), "string" == typeof a.overview && (Fc.overview = a.overview), Fc.layout ? t(Gc.slides, Fc.layout + " " + Fc.overview) : t(Gc.slides, Fc.overview)
    }

    function v(a) {
        var b = document.createElement("style");
        b.type = "text/css", b.styleSheet ? b.styleSheet.cssText = a : b.appendChild(document.createTextNode(a)), document.getElementsByTagName("head")[0].appendChild(b)
    }

    function w(a) {
        var b = a.match(/^#([0-9a-f]{3})$/i);
        if (b && b[1]) return b = b[1], {
            r: 17 * parseInt(b.charAt(0), 16),
            g: 17 * parseInt(b.charAt(1), 16),
            b: 17 * parseInt(b.charAt(2), 16)
        };
        var c = a.match(/^#([0-9a-f]{6})$/i);
        if (c && c[1]) return c = c[1], {
            r: parseInt(c.substr(0, 2), 16),
            g: parseInt(c.substr(2, 2), 16),
            b: parseInt(c.substr(4, 2), 16)
        };
        var d = a.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i);
        if (d) return {
            r: parseInt(d[1], 10),
            g: parseInt(d[2], 10),
            b: parseInt(d[3], 10)
        };
        var e = a.match(/^rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\,\s*([\d]+|[\d]*.[\d]+)\s*\)$/i);
        return e ? {
            r: parseInt(e[1], 10),
            g: parseInt(e[2], 10),
            b: parseInt(e[3], 10),
            a: parseFloat(e[4])
        } : null
    }

    function x(a) {
        return "string" == typeof a && (a = w(a)), a ? (299 * a.r + 587 * a.g + 114 * a.b) / 1e3 : null
    }

    function y(a) {
        var b = 0;
        if (a) {
            var c = 0;
            q(a.childNodes).forEach(function(a) {
                "number" == typeof a.offsetTop && a.style && ("absolute" === window.getComputedStyle(a).position && (c += 1), b = Math.max(b, a.offsetTop + a.offsetHeight))
            }), 0 === c && (b = a.offsetHeight)
        }
        return b
    }

    function z(a, b) {
        if (b = b || 0, a) {
            var c, d = a.style.height;
            return a.style.height = "0px", c = b - a.parentNode.offsetHeight, a.style.height = d + "px", c
        }
        return b
    }

    function A() {
        return /print-pdf/gi.test(window.location.search)
    }

    function B() {
        Ac.hideAddressBar && uc && (window.addEventListener("load", C, !1), window.addEventListener("orientationchange", C, !1))
    }

    function C() {
        setTimeout(function() {
            window.scrollTo(0, 1)
        }, 10)
    }

    function D(a, b) {
        var c = document.createEvent("HTMLEvents", 1, 2);
        c.initEvent(a, !0, !0), p(c, b), Gc.wrapper.dispatchEvent(c), Ac.postMessageEvents && window.parent !== window.self && window.parent.postMessage(JSON.stringify({
            namespace: "reveal",
            eventName: a,
            state: Cb()
        }), "*")
    }

    function E() {
        if (Hc.transforms3d && !("msPerspective" in document.body.style))
            for (var a = Gc.wrapper.querySelectorAll(wc + " a"), b = 0, c = a.length; c > b; b++) {
                var d = a[b];
                if (!(!d.textContent || d.querySelector("*") || d.className && d.classList.contains(d, "roll"))) {
                    var e = document.createElement("span");
                    e.setAttribute("data-title", d.text), e.innerHTML = d.innerHTML, d.classList.add("roll"), d.innerHTML = "", d.appendChild(e)
                }
            }
    }

    function F() {
        for (var a = Gc.wrapper.querySelectorAll(wc + " a.roll"), b = 0, c = a.length; c > b; b++) {
            var d = a[b],
                e = d.querySelector("span");
            e && (d.classList.remove("roll"), d.innerHTML = e.innerHTML)
        }
    }

    function G(a) {
        var b = q(document.querySelectorAll(a ? a : "a"));
        b.forEach(function(a) {
            /^(http|www)/gi.test(a.getAttribute("href")) && a.addEventListener("click", lc, !1)
        })
    }

    function H() {
        var a = q(document.querySelectorAll("a"));
        a.forEach(function(a) {
            /^(http|www)/gi.test(a.getAttribute("href")) && a.removeEventListener("click", lc, !1)
        })
    }

    function I(a) {
        K(), Gc.overlay = document.createElement("div"), Gc.overlay.classList.add("overlay"), Gc.overlay.classList.add("overlay-preview"), Gc.wrapper.appendChild(Gc.overlay), Gc.overlay.innerHTML = ["<header>", '<a class="close" href="#"><span class="icon"></span></a>', '<a class="external" href="' + a + '" target="_blank"><span class="icon"></span></a>', "</header>", '<div class="spinner"></div>', '<div class="viewport">', '<iframe src="' + a + '"></iframe>', "</div>"].join(""), Gc.overlay.querySelector("iframe").addEventListener("load", function() {
            Gc.overlay.classList.add("loaded")
        }, !1), Gc.overlay.querySelector(".close").addEventListener("click", function(a) {
            K(), a.preventDefault()
        }, !1), Gc.overlay.querySelector(".external").addEventListener("click", function() {
            K()
        }, !1), setTimeout(function() {
            Gc.overlay.classList.add("visible")
        }, 1)
    }

    function J() {
        if (Ac.help) {
            K(), Gc.overlay = document.createElement("div"), Gc.overlay.classList.add("overlay"), Gc.overlay.classList.add("overlay-help"), Gc.wrapper.appendChild(Gc.overlay);
            var a = '<p class="title">Keyboard Shortcuts</p><br/>';
            a += "<table><th>KEY</th><th>ACTION</th>";
            for (var b in Qc) a += "<tr><td>" + b + "</td><td>" + Qc[b] + "</td></tr>";
            a += "</table>", Gc.overlay.innerHTML = ["<header>", '<a class="close" href="#"><span class="icon"></span></a>', "</header>", '<div class="viewport">', '<div class="viewport-inner">' + a + "</div>", "</div>"].join(""), Gc.overlay.querySelector(".close").addEventListener("click", function(a) {
                K(), a.preventDefault()
            }, !1), setTimeout(function() {
                Gc.overlay.classList.add("visible")
            }, 1)
        }
    }

    function K() {
        Gc.overlay && (Gc.overlay.parentNode.removeChild(Gc.overlay), Gc.overlay = null)
    }

    function L() {
        if (Gc.wrapper && !A()) {
            var a = N(),
                b = 20;
            M(Ac.width, Ac.height, b), Gc.slides.style.width = a.width + "px", Gc.slides.style.height = a.height + "px", Ec = Math.min(a.presentationWidth / a.width, a.presentationHeight / a.height), Ec = Math.max(Ec, Ac.minScale), Ec = Math.min(Ec, Ac.maxScale), 1 === Ec ? (Gc.slides.style.zoom = "", Gc.slides.style.left = "", Gc.slides.style.top = "", Gc.slides.style.bottom = "", Gc.slides.style.right = "", u({
                layout: ""
            })) : !uc && /chrome/i.test(navigator.userAgent) && "undefined" != typeof Gc.slides.style.zoom ? (Gc.slides.style.left = "50%", Gc.slides.style.top = "50%", Gc.slides.style.bottom = "auto", Gc.slides.style.right = "auto", u({
                layout: "translate(-50%, -50%) scale(" + Ec + ")"
            })) : !uc && /firefox/i.test(navigator.userAgent) && "undefined" != typeof Gc.slides.style.zoom ? (Gc.slides.style.left = "50%", Gc.slides.style.top = "50%", Gc.slides.style.bottom = "auto", Gc.slides.style.right = "auto", Gc.slides.style.zoom = Ec, u({
                layout: "translate(-50%, -50%)"
            })) : (Gc.slides.style.left = "50%", Gc.slides.style.top = "50%", Gc.slides.style.bottom = "auto", Gc.slides.style.right = "auto", u({
                layout: "translate(-50%, -50%) scale(" + Ec + ")"
            }));
            for (var c = q(Gc.wrapper.querySelectorAll(wc)), d = 0, e = c.length; e > d; d++) {
                var f = c[d];
                "none" !== f.style.display && (f.style.top = Ac.center || f.classList.contains("center") ? f.classList.contains("stack") ? 0 : Math.max((a.height - y(f)) / 2 - b, 0) + "px" : "")
            }
            ib(), mb()
        }
    }

    function M(a, b) {
        q(Gc.slides.querySelectorAll("section > .stretch")).forEach(function(c) {
            var d = z(c, b);
            if (/(img|video)/gi.test(c.nodeName)) {
                var e = c.naturalWidth || c.videoWidth,
                    f = c.naturalHeight || c.videoHeight,
                    g = Math.min(a / e, d / f);
                c.style.width = e * g + "px", c.style.height = f * g + "px"
            } else c.style.width = a + "px", c.style.height = d + "px"
        })
    }

    function N(a, b) {
        var c = {
            width: Ac.width,
            height: Ac.height,
            presentationWidth: a || Gc.wrapper.offsetWidth,
            presentationHeight: b || Gc.wrapper.offsetHeight
        };
        return c.presentationWidth -= c.presentationHeight * Ac.margin, c.presentationHeight -= c.presentationHeight * Ac.margin, "string" == typeof c.width && /%$/.test(c.width) && (c.width = parseInt(c.width, 10) / 100 * c.presentationWidth), "string" == typeof c.height && /%$/.test(c.height) && (c.height = parseInt(c.height, 10) / 100 * c.presentationHeight), c
    }

    function O(a, b) {
        "object" == typeof a && "function" == typeof a.setAttribute && a.setAttribute("data-previous-indexv", b || 0)
    }

    function P(a) {
        if ("object" == typeof a && "function" == typeof a.setAttribute && a.classList.contains("stack")) {
            var b = a.hasAttribute("data-start-indexv") ? "data-start-indexv" : "data-previous-indexv";
            return parseInt(a.getAttribute(b) || 0, 10)
        }
        return 0
    }

    function Q() {
        Ac.overview && !V() && (Cc = !0, Gc.wrapper.classList.add("overview"), Gc.wrapper.classList.remove("overview-deactivating"), Hc.overviewTransitions && setTimeout(function() {
            Gc.wrapper.classList.add("overview-animated")
        }, 1), Jb(), Gc.slides.appendChild(Gc.background), q(Gc.wrapper.querySelectorAll(wc)).forEach(function(a) {
            a.classList.contains("stack") || a.addEventListener("click", kc, !0)
        }), hb(), R(), S(), L(), D("overviewshown", {
            indexh: pc,
            indexv: qc,
            currentSlide: sc
        }))
    }

    function R() {
        var a = 70,
            b = Ac.width + a,
            c = Ac.height + a;
        Ac.rtl && (b = -b), q(Gc.wrapper.querySelectorAll(xc)).forEach(function(a, d) {
            a.setAttribute("data-index-h", d), t(a, "translate3d(" + d * b + "px, 0, 0)"), a.classList.contains("stack") && q(a.querySelectorAll("section")).forEach(function(a, b) {
                a.setAttribute("data-index-h", d), a.setAttribute("data-index-v", b), t(a, "translate3d(0, " + b * c + "px, 0)")
            })
        }), q(Gc.background.childNodes).forEach(function(a, d) {
            t(a, "translate3d(" + d * b + "px, 0, 0)"), q(a.querySelectorAll(".slide-background")).forEach(function(a, b) {
                t(a, "translate3d(0, " + b * c + "px, 0)")
            })
        })
    }

    function S() {
        var a = 70,
            b = Ac.width + a,
            c = Ac.height + a;
        Ac.rtl && (b = -b), u({
            overview: ["translateX(" + -pc * b + "px)", "translateY(" + -qc * c + "px)", "translateZ(" + (window.innerWidth < 400 ? -1e3 : -2500) + "px)"].join(" ")
        })
    }

    function T() {
        Ac.overview && (Cc = !1, Gc.wrapper.classList.remove("overview"), Gc.wrapper.classList.remove("overview-animated"), Gc.wrapper.classList.add("overview-deactivating"), setTimeout(function() {
            Gc.wrapper.classList.remove("overview-deactivating")
        }, 1), Gc.wrapper.appendChild(Gc.background), q(Gc.wrapper.querySelectorAll(wc)).forEach(function(a) {
            t(a, ""), a.removeEventListener("click", kc, !0)
        }), q(Gc.background.querySelectorAll(".slide-background")).forEach(function(a) {
            t(a, "")
        }), u({
            overview: ""
        }), cb(pc, qc), L(), Ib(), D("overviewhidden", {
            indexh: pc,
            indexv: qc,
            currentSlide: sc
        }))
    }

    function U(a) {
        "boolean" == typeof a ? a ? Q() : T() : V() ? T() : Q()
    }

    function V() {
        return Cc
    }

    function W(a) {
        return a = a ? a : sc, a && a.parentNode && !!a.parentNode.nodeName.match(/section/i)
    }

    function X() {
        var a = document.body,
            b = a.requestFullScreen || a.webkitRequestFullscreen || a.webkitRequestFullScreen || a.mozRequestFullScreen || a.msRequestFullscreen;
        b && b.apply(a)
    }

    function Y() {
        if (Ac.pause) {
            var a = Gc.wrapper.classList.contains("paused");
            Jb(), Gc.wrapper.classList.add("paused"), a === !1 && D("paused")
        }
    }

    function Z() {
        var a = Gc.wrapper.classList.contains("paused");
        Gc.wrapper.classList.remove("paused"), Ib(), a && D("resumed")
    }

    function $(a) {
        "boolean" == typeof a ? a ? Y() : Z() : _() ? Z() : Y()
    }

    function _() {
        return Gc.wrapper.classList.contains("paused")
    }

    function ab(a) {
        "boolean" == typeof a ? a ? Lb() : Kb() : Oc ? Lb() : Kb()
    }

    function bb() {
        return !(!Lc || Oc)
    }

    function cb(a, b, c, d) {
        rc = sc;
        var e = Gc.wrapper.querySelectorAll(xc);
        void 0 !== b || V() || (b = P(e[a])), rc && rc.parentNode && rc.parentNode.classList.contains("stack") && O(rc.parentNode, qc);
        var f = Dc.concat();
        Dc.length = 0;
        var g = pc || 0,
            h = qc || 0;
        pc = gb(xc, void 0 === a ? pc : a), qc = gb(yc, void 0 === b ? qc : b), hb(), L();
        a: for (var i = 0, j = Dc.length; j > i; i++) {
            for (var k = 0; k < f.length; k++)
                if (f[k] === Dc[i]) {
                    f.splice(k, 1);
                    continue a
                }
            document.documentElement.classList.add(Dc[i]), D(Dc[i])
        }
        for (; f.length;) document.documentElement.classList.remove(f.pop());
        V() && S();
        var l = e[pc],
            m = l.querySelectorAll("section");
        sc = m[qc] || l, "undefined" != typeof c && Fb(c);
        var n = pc !== g || qc !== h;
        n ? D("slidechanged", {
            indexh: pc,
            indexv: qc,
            previousSlide: rc,
            currentSlide: sc,
            origin: d
        }) : rc = null, rc && (rc.classList.remove("present"), rc.setAttribute("aria-hidden", "true"), Gc.wrapper.querySelector(zc).classList.contains("present") && setTimeout(function() {
            var a, b = q(Gc.wrapper.querySelectorAll(xc + ".stack"));
            for (a in b) b[a] && O(b[a], 0)
        }, 0)), (n || !rc) && (tb(rc), sb(sc)), Gc.statusDiv.textContent = sc.textContent, kb(), ib(), lb(), mb(), jb(), xb(), Ib()
    }

    function db() {
        o(), n(), L(), Lc = Ac.autoSlide, Ib(), j(), xb(), fb(), kb(), ib(), lb(!0), jb(), hb(), rb(), V() && R()
    }

    function eb() {
        var a = q(Gc.wrapper.querySelectorAll(xc));
        a.forEach(function(a) {
            var b = q(a.querySelectorAll("section"));
            b.forEach(function(a, b) {
                b > 0 && (a.classList.remove("present"), a.classList.remove("past"), a.classList.add("future"), a.setAttribute("aria-hidden", "true"))
            })
        })
    }

    function fb() {
        var a = q(Gc.wrapper.querySelectorAll(xc));
        a.forEach(function(a) {
            var b = q(a.querySelectorAll("section"));
            b.forEach(function(a) {
                Eb(a.querySelectorAll(".fragment"))
            }), 0 === b.length && Eb(a.querySelectorAll(".fragment"))
        })
    }

    function gb(a, b) {
        var c = q(Gc.wrapper.querySelectorAll(a)),
            d = c.length,
            e = A();
        if (d) {
            Ac.loop && (b %= d, 0 > b && (b = d + b)), b = Math.max(Math.min(b, d - 1), 0);
            for (var f = 0; d > f; f++) {
                var g = c[f],
                    h = Ac.rtl && !W(g);
                if (g.classList.remove("past"), g.classList.remove("present"), g.classList.remove("future"), g.setAttribute("hidden", ""), g.setAttribute("aria-hidden", "true"), g.querySelector("section") && g.classList.add("stack"), e) g.classList.add("present");
                else if (b > f) {
                    if (g.classList.add(h ? "future" : "past"), Ac.fragments)
                        for (var i = q(g.querySelectorAll(".fragment")); i.length;) {
                            var j = i.pop();
                            j.classList.add("visible"), j.classList.remove("current-fragment")
                        }
                } else if (f > b && (g.classList.add(h ? "past" : "future"), Ac.fragments))
                    for (var k = q(g.querySelectorAll(".fragment.visible")); k.length;) {
                        var l = k.pop();
                        l.classList.remove("visible"), l.classList.remove("current-fragment")
                    }
            }
            c[b].classList.add("present"), c[b].removeAttribute("hidden"), c[b].removeAttribute("aria-hidden");
            var m = c[b].getAttribute("data-state");
            m && (Dc = Dc.concat(m.split(" ")))
        } else b = 0;
        return b
    }

    function hb() {
        var a, b, c = q(Gc.wrapper.querySelectorAll(xc)),
            d = c.length;
        if (d && "undefined" != typeof pc) {
            var e = V() ? 10 : Ac.viewDistance;
            uc && (e = V() ? 6 : 2), A() && (e = Number.MAX_VALUE);
            for (var f = 0; d > f; f++) {
                var g = c[f],
                    h = q(g.querySelectorAll("section")),
                    i = h.length;
                if (a = Math.abs((pc || 0) - f) || 0, Ac.loop && (a = Math.abs(((pc || 0) - f) % (d - e)) || 0), e > a ? nb(g) : ob(g), i)
                    for (var j = P(g), k = 0; i > k; k++) {
                        var l = h[k];
                        b = Math.abs(f === (pc || 0) ? (qc || 0) - k : k - j), e > a + b ? nb(l) : ob(l)
                    }
            }
        }
    }

    function ib() {
        Ac.progress && Gc.progressbar && (Gc.progressbar.style.width = ub() * Gc.wrapper.offsetWidth + "px")
    }

    function jb() {
        if (Ac.slideNumber && Gc.slideNumber) {
            var a = "c";
            "string" == typeof Ac.slideNumber && (a = Ac.slideNumber);
            var b = zb();
            Gc.slideNumber.innerHTML = a.replace(/h/g, pc).replace(/v/g, qc).replace(/c/g, Math.round(ub() * b) + 1).replace(/t/g, b + 1)
        }
    }

    function kb() {
        var a = pb(),
            b = qb();
        Gc.controlsLeft.concat(Gc.controlsRight).concat(Gc.controlsUp).concat(Gc.controlsDown).concat(Gc.controlsPrev).concat(Gc.controlsNext).forEach(function(a) {
            a.classList.remove("enabled"), a.classList.remove("fragmented")
        }), a.left && Gc.controlsLeft.forEach(function(a) {
            a.classList.add("enabled")
        }), a.right && Gc.controlsRight.forEach(function(a) {
            a.classList.add("enabled")
        }), a.up && Gc.controlsUp.forEach(function(a) {
            a.classList.add("enabled")
        }), a.down && Gc.controlsDown.forEach(function(a) {
            a.classList.add("enabled")
        }), (a.left || a.up) && Gc.controlsPrev.forEach(function(a) {
            a.classList.add("enabled")
        }), (a.right || a.down) && Gc.controlsNext.forEach(function(a) {
            a.classList.add("enabled")
        }), sc && (b.prev && Gc.controlsPrev.forEach(function(a) {
            a.classList.add("fragmented", "enabled")
        }), b.next && Gc.controlsNext.forEach(function(a) {
            a.classList.add("fragmented", "enabled")
        }), W(sc) ? (b.prev && Gc.controlsUp.forEach(function(a) {
            a.classList.add("fragmented", "enabled")
        }), b.next && Gc.controlsDown.forEach(function(a) {
            a.classList.add("fragmented", "enabled")
        })) : (b.prev && Gc.controlsLeft.forEach(function(a) {
            a.classList.add("fragmented", "enabled")
        }), b.next && Gc.controlsRight.forEach(function(a) {
            a.classList.add("fragmented", "enabled")
        })))
    }

    function lb(a) {
        var b = null,
            c = Ac.rtl ? "future" : "past",
            d = Ac.rtl ? "past" : "future";
        if (q(Gc.background.childNodes).forEach(function(e, f) {
                e.classList.remove("past"), e.classList.remove("present"), e.classList.remove("future"), pc > f ? e.classList.add(c) : f > pc ? e.classList.add(d) : (e.classList.add("present"), b = e), (a || f === pc) && q(e.querySelectorAll(".slide-background")).forEach(function(a, c) {
                    a.classList.remove("past"), a.classList.remove("present"), a.classList.remove("future"), qc > c ? a.classList.add("past") : c > qc ? a.classList.add("future") : (a.classList.add("present"), f === pc && (b = a))
                })
            }), tc) {
            var e = tc.querySelector("video");
            e && e.pause()
        }
        if (b) {
            var f = b.querySelector("video");
            f && (f.currentTime = 0, f.play());
            var g = b.style.backgroundImage || "";
            /\.gif/i.test(g) && (b.style.backgroundImage = "", window.getComputedStyle(b).opacity, b.style.backgroundImage = g);
            var h = tc ? tc.getAttribute("data-background-hash") : null,
                i = b.getAttribute("data-background-hash");
            i && i === h && b !== tc && Gc.background.classList.add("no-transition"), tc = b
        }
        sc && ["has-light-background", "has-dark-background"].forEach(function(a) {
            sc.classList.contains(a) ? Gc.wrapper.classList.add(a) : Gc.wrapper.classList.remove(a)
        }), setTimeout(function() {
            Gc.background.classList.remove("no-transition")
        }, 1)
    }

    function mb() {
        if (Ac.parallaxBackgroundImage) {
            var a, b, c = Gc.wrapper.querySelectorAll(xc),
                d = Gc.wrapper.querySelectorAll(yc),
                e = Gc.background.style.backgroundSize.split(" ");
            1 === e.length ? a = b = parseInt(e[0], 10) : (a = parseInt(e[0], 10), b = parseInt(e[1], 10));
            var f, g, h = Gc.background.offsetWidth,
                i = c.length;
            f = "number" == typeof Ac.parallaxBackgroundHorizontal ? Ac.parallaxBackgroundHorizontal : (a - h) / (i - 1), g = f * pc * -1;
            var j, k, l = Gc.background.offsetHeight,
                m = d.length;
            j = "number" == typeof Ac.parallaxBackgroundVertical ? Ac.parallaxBackgroundVertical : (b - l) / (m - 1), k = m > 0 ? j * qc * 1 : 0, Gc.background.style.backgroundPosition = g + "px " + -k + "px"
        }
    }

    function nb(a) {
        a.style.display = "block", q(a.querySelectorAll("img[data-src], video[data-src], audio[data-src], iframe[data-src]")).forEach(function(a) {
            a.setAttribute("src", a.getAttribute("data-src")), a.removeAttribute("data-src")
        }), q(a.querySelectorAll("video, audio")).forEach(function(a) {
            var b = 0;
            q(a.querySelectorAll("source[data-src]")).forEach(function(a) {
                a.setAttribute("src", a.getAttribute("data-src")), a.removeAttribute("data-src"), b += 1
            }), b > 0 && a.load()
        });
        var b = yb(a),
            c = Bb(b.h, b.v);
        if (c && (c.style.display = "block", c.hasAttribute("data-loaded") === !1)) {
            c.setAttribute("data-loaded", "true");
            var d = a.getAttribute("data-background-image"),
                e = a.getAttribute("data-background-video"),
                f = a.hasAttribute("data-background-video-loop"),
                g = a.getAttribute("data-background-iframe");
            if (d) c.style.backgroundImage = "url(" + d + ")";
            else if (e && !vb()) {
                var h = document.createElement("video");
                f && h.setAttribute("loop", ""), e.split(",").forEach(function(a) {
                    h.innerHTML += '<source src="' + a + '">'
                }), c.appendChild(h)
            } else if (g) {
                var i = document.createElement("iframe");
                i.setAttribute("src", g), i.style.width = "100%", i.style.height = "100%", i.style.maxHeight = "100%", i.style.maxWidth = "100%", c.appendChild(i)
            }
        }
    }

    function ob(a) {
        a.style.display = "none";
        var b = yb(a),
            c = Bb(b.h, b.v);
        c && (c.style.display = "none")
    }

    function pb() {
        var a = Gc.wrapper.querySelectorAll(xc),
            b = Gc.wrapper.querySelectorAll(yc),
            c = {
                left: pc > 0 || Ac.loop,
                right: pc < a.length - 1 || Ac.loop,
                up: qc > 0,
                down: qc < b.length - 1
            };
        if (Ac.rtl) {
            var d = c.left;
            c.left = c.right, c.right = d
        }
        return c
    }

    function qb() {
        if (sc && Ac.fragments) {
            var a = sc.querySelectorAll(".fragment"),
                b = sc.querySelectorAll(".fragment:not(.visible)");
            return {
                prev: a.length - b.length > 0,
                next: !!b.length
            }
        }
        return {
            prev: !1,
            next: !1
        }
    }

    function rb() {
        q(Gc.slides.querySelectorAll('iframe[src*="youtube.com/embed/"]')).forEach(function(a) {
            var b = a.getAttribute("src");
            /enablejsapi\=1/gi.test(b) || a.setAttribute("src", b + (/\?/.test(b) ? "&" : "?") + "enablejsapi=1")
        }), q(Gc.slides.querySelectorAll('iframe[src*="player.vimeo.com/"]')).forEach(function(a) {
            var b = a.getAttribute("src");
            /api\=1/gi.test(b) || a.setAttribute("src", b + (/\?/.test(b) ? "&" : "?") + "api=1")
        })
    }

    function sb(a) {
        a && !vb() && (q(a.querySelectorAll('img[src$=".gif"]')).forEach(function(a) {
            a.setAttribute("src", a.getAttribute("src"))
        }), q(a.querySelectorAll("video, audio")).forEach(function(a) {
            a.hasAttribute("data-autoplay") && a.play()
        }), q(a.querySelectorAll("iframe")).forEach(function(a) {
            a.contentWindow.postMessage("slide:start", "*")
        }), q(a.querySelectorAll('iframe[src*="youtube.com/embed/"]')).forEach(function(a) {
            a.hasAttribute("data-autoplay") && a.contentWindow.postMessage('{"event":"command","func":"playVideo","args":""}', "*")
        }), q(a.querySelectorAll('iframe[src*="player.vimeo.com/"]')).forEach(function(a) {
            a.hasAttribute("data-autoplay") && a.contentWindow.postMessage('{"method":"play"}', "*")
        }))
    }

    function tb(a) {
        a && a.parentNode && (q(a.querySelectorAll("video, audio")).forEach(function(a) {
            a.hasAttribute("data-ignore") || a.pause()
        }), q(a.querySelectorAll("iframe")).forEach(function(a) {
            a.contentWindow.postMessage("slide:stop", "*")
        }), q(a.querySelectorAll('iframe[src*="youtube.com/embed/"]')).forEach(function(a) {
            a.hasAttribute("data-ignore") || "function" != typeof a.contentWindow.postMessage || a.contentWindow.postMessage('{"event":"command","func":"pauseVideo","args":""}', "*")
        }), q(a.querySelectorAll('iframe[src*="player.vimeo.com/"]')).forEach(function(a) {
            a.hasAttribute("data-ignore") || "function" != typeof a.contentWindow.postMessage || a.contentWindow.postMessage('{"method":"pause"}', "*")
        }))
    }

    function ub() {
        var a = q(Gc.wrapper.querySelectorAll(xc)),
            b = zb(),
            c = 0;
        a: for (var d = 0; d < a.length; d++) {
            for (var e = a[d], f = q(e.querySelectorAll("section")), g = 0; g < f.length; g++) {
                if (f[g].classList.contains("present")) break a;
                c++
            }
            if (e.classList.contains("present")) break;
            e.classList.contains("stack") === !1 && c++
        }
        if (sc) {
            var h = sc.querySelectorAll(".fragment");
            if (h.length > 0) {
                var i = sc.querySelectorAll(".fragment.visible"),
                    j = .9;
                c += i.length / h.length * j
            }
        }
        return c / (b - 1)
    }

    function vb() {
        return !!window.location.search.match(/receiver/gi)
    }

    function wb() {
        var a = window.location.hash,
            b = a.slice(2).split("/"),
            c = a.replace(/#|\//gi, "");
        if (isNaN(parseInt(b[0], 10)) && c.length) {
            var d;
            if (/^[a-zA-Z][\w:.-]*$/.test(c) && (d = document.getElementById(c)), d) {
                var e = oc.getIndices(d);
                cb(e.h, e.v)
            } else cb(pc || 0, qc || 0)
        } else {
            var f = parseInt(b[0], 10) || 0,
                g = parseInt(b[1], 10) || 0;
            (f !== pc || g !== qc) && cb(f, g)
        }
    }

    function xb(a) {
        if (Ac.history)
            if (clearTimeout(Jc), "number" == typeof a) Jc = setTimeout(xb, a);
            else if (sc) {
            var b = "/",
                c = sc.getAttribute("id");
            c && (c = c.toLowerCase(), c = c.replace(/[^a-zA-Z0-9\-\_\:\.]/g, "")), "string" == typeof c && c.length ? b = "/" + c : ((pc > 0 || qc > 0) && (b += pc), qc > 0 && (b += "/" + qc)), window.location.hash = b
        }
    }

    function yb(a) {
        var b, c = pc,
            d = qc;
        if (a) {
            var e = W(a),
                f = e ? a.parentNode : a,
                g = q(Gc.wrapper.querySelectorAll(xc));
            c = Math.max(g.indexOf(f), 0), d = void 0, e && (d = Math.max(q(a.parentNode.querySelectorAll("section")).indexOf(a), 0))
        }
        if (!a && sc) {
            var h = sc.querySelectorAll(".fragment").length > 0;
            if (h) {
                var i = sc.querySelector(".current-fragment");
                b = i && i.hasAttribute("data-fragment-index") ? parseInt(i.getAttribute("data-fragment-index"), 10) : sc.querySelectorAll(".fragment.visible").length - 1
            }
        }
        return {
            h: c,
            v: d,
            f: b
        }
    }

    function zb() {
        return Gc.wrapper.querySelectorAll(wc + ":not(.stack)").length
    }

    function Ab(a, b) {
        var c = Gc.wrapper.querySelectorAll(xc)[a],
            d = c && c.querySelectorAll("section");
        return d && d.length && "number" == typeof b ? d ? d[b] : void 0 : c
    }

    function Bb(a, b) {
        if (A()) {
            var c = Ab(a, b);
            if (c) {
                var d = c.querySelector(".slide-background");
                if (d && d.parentNode === c) return d
            }
            return void 0
        }
        var e = Gc.wrapper.querySelectorAll(".backgrounds>.slide-background")[a],
            f = e && e.querySelectorAll(".slide-background");
        return f && f.length && "number" == typeof b ? f ? f[b] : void 0 : e
    }

    function Cb() {
        var a = yb();
        return {
            indexh: a.h,
            indexv: a.v,
            indexf: a.f,
            paused: _(),
            overview: V()
        }
    }

    function Db(a) {
        if ("object" == typeof a) {
            cb(r(a.indexh), r(a.indexv), r(a.indexf));
            var b = r(a.paused),
                c = r(a.overview);
            "boolean" == typeof b && b !== _() && $(b), "boolean" == typeof c && c !== V() && U(c)
        }
    }

    function Eb(a) {
        a = q(a);
        var b = [],
            c = [],
            d = [];
        a.forEach(function(a) {
            if (a.hasAttribute("data-fragment-index")) {
                var d = parseInt(a.getAttribute("data-fragment-index"), 10);
                b[d] || (b[d] = []), b[d].push(a)
            } else c.push([a])
        }), b = b.concat(c);
        var e = 0;
        return b.forEach(function(a) {
            a.forEach(function(a) {
                d.push(a), a.setAttribute("data-fragment-index", e)
            }), e++
        }), d
    }

    function Fb(a, b) {
        if (sc && Ac.fragments) {
            var c = Eb(sc.querySelectorAll(".fragment"));
            if (c.length) {
                if ("number" != typeof a) {
                    var d = Eb(sc.querySelectorAll(".fragment.visible")).pop();
                    a = d ? parseInt(d.getAttribute("data-fragment-index") || 0, 10) : -1
                }
                "number" == typeof b && (a += b);
                var e = [],
                    f = [];
                return q(c).forEach(function(b, c) {
                    b.hasAttribute("data-fragment-index") && (c = parseInt(b.getAttribute("data-fragment-index"), 10)), a >= c ? (b.classList.contains("visible") || e.push(b), b.classList.add("visible"), b.classList.remove("current-fragment"), Gc.statusDiv.textContent = b.textContent, c === a && b.classList.add("current-fragment")) : (b.classList.contains("visible") && f.push(b), b.classList.remove("visible"), b.classList.remove("current-fragment"))
                }), f.length && D("fragmenthidden", {
                    fragment: f[0],
                    fragments: f
                }), e.length && D("fragmentshown", {
                    fragment: e[0],
                    fragments: e
                }), kb(), ib(), !(!e.length && !f.length)
            }
        }
        return !1
    }

    function Gb() {
        return Fb(null, 1)
    }

    function Hb() {
        return Fb(null, -1)
    }

    function Ib() {
        if (Jb(), sc) {
            var a = sc.querySelector(".current-fragment"),
                b = a ? a.getAttribute("data-autoslide") : null,
                c = sc.parentNode ? sc.parentNode.getAttribute("data-autoslide") : null,
                d = sc.getAttribute("data-autoslide");
            Lc = b ? parseInt(b, 10) : d ? parseInt(d, 10) : c ? parseInt(c, 10) : Ac.autoSlide, q(sc.querySelectorAll("video, audio")).forEach(function(a) {
                a.hasAttribute("data-autoplay") && Lc && 1e3 * a.duration > Lc && (Lc = 1e3 * a.duration + 1e3)
            }), !Lc || Oc || _() || V() || oc.isLastSlide() && !qb().next && Ac.loop !== !0 || (Mc = setTimeout(Rb, Lc), Nc = Date.now()), vc && vc.setPlaying(-1 !== Mc)
        }
    }

    function Jb() {
        clearTimeout(Mc), Mc = -1
    }

    function Kb() {
        Lc && !Oc && (Oc = !0, D("autoslidepaused"), clearTimeout(Mc), vc && vc.setPlaying(!1))
    }

    function Lb() {
        Lc && Oc && (Oc = !1, D("autoslideresumed"), Ib())
    }

    function Mb() {
        Ac.rtl ? (V() || Gb() === !1) && pb().left && cb(pc + 1) : (V() || Hb() === !1) && pb().left && cb(pc - 1)
    }

    function Nb() {
        Ac.rtl ? (V() || Hb() === !1) && pb().right && cb(pc - 1) : (V() || Gb() === !1) && pb().right && cb(pc + 1)
    }

    function Ob() {
        (V() || Hb() === !1) && pb().up && cb(pc, qc - 1)
    }

    function Pb() {
        (V() || Gb() === !1) && pb().down && cb(pc, qc + 1)
    }

    function Qb() {
        if (Hb() === !1)
            if (pb().up) Ob();
            else {
                var a;
                if (a = Ac.rtl ? q(Gc.wrapper.querySelectorAll(xc + ".future")).pop() : q(Gc.wrapper.querySelectorAll(xc + ".past")).pop()) {
                    var b = a.querySelectorAll("section").length - 1 || void 0,
                        c = pc - 1;
                    cb(c, b)
                }
            }
    }

    function Rb() {
        Gb() === !1 && (pb().down ? Pb() : Ac.rtl ? Mb() : Nb()), Ib()
    }

    function Sb() {
        Ac.autoSlideStoppable && Kb()
    }

    function Tb(a) {
        a.shiftKey && 63 === a.charCode && (Gc.overlay ? K() : J(!0))
    }

    function Ub(a) {
        if ("function" == typeof Ac.keyboardCondition && Ac.keyboardCondition() === !1) return !0;
        var b = Oc;
        Sb(a);
        var c = document.activeElement && "inherit" !== document.activeElement.contentEditable,
            d = document.activeElement && document.activeElement.tagName && /input|textarea/i.test(document.activeElement.tagName);
        if (!(c || d || a.shiftKey && 32 !== a.keyCode || a.altKey || a.ctrlKey || a.metaKey)) {
            if (_() && -1 === [66, 190, 191].indexOf(a.keyCode)) return !1;
            var e = !1;
            if ("object" == typeof Ac.keyboard)
                for (var f in Ac.keyboard)
                    if (parseInt(f, 10) === a.keyCode) {
                        var g = Ac.keyboard[f];
                        "function" == typeof g ? g.apply(null, [a]) : "string" == typeof g && "function" == typeof oc[g] && oc[g].call(), e = !0
                    }
            if (e === !1) switch (e = !0, a.keyCode) {
                case 80:
                case 33:
                    Qb();
                    break;
                case 78:
                case 34:
                    Rb();
                    break;
                case 72:
                case 37:
                    Mb();
                    break;
                case 76:
                case 39:
                    Nb();
                    break;
                case 75:
                case 38:
                    Ob();
                    break;
                case 74:
                case 40:
                    Pb();
                    break;
                case 36:
                    cb(0);
                    break;
                case 35:
                    cb(Number.MAX_VALUE);
                    break;
                case 32:
                    V() ? T() : a.shiftKey ? Qb() : Rb();
                    break;
                case 13:
                    V() ? T() : e = !1;
                    break;
                case 58:
                case 59:
                case 66:
                case 190:
                case 191:
                    $();
                    break;
                case 70:
                    X();
                    break;
                case 65:
                    Ac.autoSlideStoppable && ab(b);
                    break;
                default:
                    e = !1
            }
            e ? a.preventDefault && a.preventDefault() : 27 !== a.keyCode && 79 !== a.keyCode || !Hc.transforms3d || (Gc.overlay ? K() : U(), a.preventDefault && a.preventDefault()), Ib()
        }
    }

    function Vb(a) {
        Pc.startX = a.touches[0].clientX, Pc.startY = a.touches[0].clientY, Pc.startCount = a.touches.length, 2 === a.touches.length && Ac.overview && (Pc.startSpan = s({
            x: a.touches[1].clientX,
            y: a.touches[1].clientY
        }, {
            x: Pc.startX,
            y: Pc.startY
        }))
    }

    function Wb(a) {
        if (Pc.captured) navigator.userAgent.match(/android/gi) && a.preventDefault();
        else {
            Sb(a);
            var b = a.touches[0].clientX,
                c = a.touches[0].clientY;
            if (2 === a.touches.length && 2 === Pc.startCount && Ac.overview) {
                var d = s({
                    x: a.touches[1].clientX,
                    y: a.touches[1].clientY
                }, {
                    x: Pc.startX,
                    y: Pc.startY
                });
                Math.abs(Pc.startSpan - d) > Pc.threshold && (Pc.captured = !0, d < Pc.startSpan ? Q() : T()), a.preventDefault()
            } else if (1 === a.touches.length && 2 !== Pc.startCount) {
                var e = b - Pc.startX,
                    f = c - Pc.startY;
                e > Pc.threshold && Math.abs(e) > Math.abs(f) ? (Pc.captured = !0, Mb()) : e < -Pc.threshold && Math.abs(e) > Math.abs(f) ? (Pc.captured = !0, Nb()) : f > Pc.threshold ? (Pc.captured = !0, Ob()) : f < -Pc.threshold && (Pc.captured = !0, Pb()), Ac.embedded ? (Pc.captured || W(sc)) && a.preventDefault() : a.preventDefault()
            }
        }
    }

    function Xb() {
        Pc.captured = !1
    }

    function Yb(a) {
        (a.pointerType === a.MSPOINTER_TYPE_TOUCH || "touch" === a.pointerType) && (a.touches = [{
            clientX: a.clientX,
            clientY: a.clientY
        }], Vb(a))
    }

    function Zb(a) {
        (a.pointerType === a.MSPOINTER_TYPE_TOUCH || "touch" === a.pointerType) && (a.touches = [{
            clientX: a.clientX,
            clientY: a.clientY
        }], Wb(a))
    }

    function $b(a) {
        (a.pointerType === a.MSPOINTER_TYPE_TOUCH || "touch" === a.pointerType) && (a.touches = [{
            clientX: a.clientX,
            clientY: a.clientY
        }], Xb(a))
    }

    function _b(a) {
        if (Date.now() - Ic > 600) {
            Ic = Date.now();
            var b = a.detail || -a.wheelDelta;
            b > 0 ? Rb() : Qb()
        }
    }

    function ac(a) {
        Sb(a), a.preventDefault();
        var b = q(Gc.wrapper.querySelectorAll(xc)).length,
            c = Math.floor(a.clientX / Gc.wrapper.offsetWidth * b);
        Ac.rtl && (c = b - c), cb(c)
    }

    function bc(a) {
        a.preventDefault(), Sb(), Mb()
    }

    function cc(a) {
        a.preventDefault(), Sb(), Nb()
    }

    function dc(a) {
        a.preventDefault(), Sb(), Ob()
    }

    function ec(a) {
        a.preventDefault(), Sb(), Pb()
    }

    function fc(a) {
        a.preventDefault(), Sb(), Qb()
    }

    function gc(a) {
        a.preventDefault(), Sb(), Rb()
    }

    function hc() {
        wb()
    }

    function ic() {
        L()
    }

    function jc() {
        var a = document.webkitHidden || document.msHidden || document.hidden;
        a === !1 && document.activeElement !== document.body && ("function" == typeof document.activeElement.blur && document.activeElement.blur(), document.body.focus())
    }

    function kc(a) {
        if (Kc && V()) {
            a.preventDefault();
            for (var b = a.target; b && !b.nodeName.match(/section/gi);) b = b.parentNode;
            if (b && !b.classList.contains("disabled") && (T(), b.nodeName.match(/section/gi))) {
                var c = parseInt(b.getAttribute("data-index-h"), 10),
                    d = parseInt(b.getAttribute("data-index-v"), 10);
                cb(c, d)
            }
        }
    }

    function lc(a) {
        if (a.currentTarget && a.currentTarget.hasAttribute("href")) {
            var b = a.currentTarget.getAttribute("href");
            b && (I(b), a.preventDefault())
        }
    }

    function mc() {
        oc.isLastSlide() && Ac.loop === !1 ? (cb(0, 0), Lb()) : Oc ? Lb() : Kb()
    }

    function nc(a, b) {
        this.diameter = 50, this.thickness = 3, this.playing = !1, this.progress = 0, this.progressOffset = 1, this.container = a, this.progressCheck = b, this.canvas = document.createElement("canvas"), this.canvas.className = "playback", this.canvas.width = this.diameter, this.canvas.height = this.diameter, this.context = this.canvas.getContext("2d"), this.container.appendChild(this.canvas), this.render()
    }
    var oc, pc, qc, rc, sc, tc, uc, vc, wc = ".slides section",
        xc = ".slides>section",
        yc = ".slides>section.present>section",
        zc = ".slides>section:first-of-type",
        Ac = {
            width: 960,
            height: 700,
            margin: .1,
            minScale: .2,
            maxScale: 1.5,
            controls: !0,
            progress: !0,
            slideNumber: !1,
            history: !1,
            keyboard: !0,
            keyboardCondition: null,
            overview: !0,
            center: !0,
            touch: !0,
            loop: !1,
            rtl: !1,
            fragments: 1,
            embedded: !1,
            help: !0,
            pause: !0,
            autoSlide: 0,
            autoSlideStoppable: !0,
            mouseWheel: !1,
            rollingLinks: !1,
            hideAddressBar: !0,
            previewLinks: !1,
            postMessage: !0,
            postMessageEvents: !1,
            focusBodyOnPageVisibilityChange: !0,
            transition: "slide",
            transitionSpeed: "default",
            backgroundTransition: "fade",
            parallaxBackgroundImage: "",
            parallaxBackgroundSize: "",
            parallaxBackgroundHorizontal: null,
            parallaxBackgroundVertical: null,
            viewDistance: 3,
            dependencies: []
        },
        Bc = !1,
        Cc = !1,
        Dc = [],
        Ec = 1,
        Fc = {
            layout: "",
            overview: ""
        },
        Gc = {},
        Hc = {},
        Ic = 0,
        Jc = 0,
        Kc = !1,
        Lc = 0,
        Mc = 0,
        Nc = -1,
        Oc = !1,
        Pc = {
            startX: 0,
            startY: 0,
            startSpan: 0,
            startCount: 0,
            captured: !1,
            threshold: 40
        },
        Qc = {
            "N  ,  SPACE": "Next slide",
            P: "Previous slide",
            "&#8592;  ,  H": "Navigate left",
            "&#8594;  ,  L": "Navigate right",
            "&#8593;  ,  K": "Navigate up",
            "&#8595;  ,  J": "Navigate down",
            Home: "First slide",
            End: "Last slide",
            "B  ,  .": "Pause",
            F: "Fullscreen",
            "ESC, O": "Slide overview"
        };
    return nc.prototype.setPlaying = function(a) {
        var b = this.playing;
        this.playing = a, !b && this.playing ? this.animate() : this.render()
    }, nc.prototype.animate = function() {
        var a = this.progress;
        this.progress = this.progressCheck(), a > .8 && this.progress < .2 && (this.progressOffset = this.progress), this.render(), this.playing && Hc.requestAnimationFrameMethod.call(window, this.animate.bind(this))
    }, nc.prototype.render = function() {
        var a = this.playing ? this.progress : 0,
            b = this.diameter / 2 - this.thickness,
            c = this.diameter / 2,
            d = this.diameter / 2,
            e = 14;
        this.progressOffset += .1 * (1 - this.progressOffset);
        var f = -Math.PI / 2 + 2 * a * Math.PI,
            g = -Math.PI / 2 + 2 * this.progressOffset * Math.PI;
        this.context.save(), this.context.clearRect(0, 0, this.diameter, this.diameter), this.context.beginPath(), this.context.arc(c, d, b + 2, 0, 2 * Math.PI, !1), this.context.fillStyle = "rgba( 0, 0, 0, 0.4 )", this.context.fill(), this.context.beginPath(), this.context.arc(c, d, b, 0, 2 * Math.PI, !1), this.context.lineWidth = this.thickness, this.context.strokeStyle = "#666", this.context.stroke(), this.playing && (this.context.beginPath(), this.context.arc(c, d, b, g, f, !1), this.context.lineWidth = this.thickness, this.context.strokeStyle = "#fff", this.context.stroke()), this.context.translate(c - e / 2, d - e / 2), this.playing ? (this.context.fillStyle = "#fff", this.context.fillRect(0, 0, e / 2 - 2, e), this.context.fillRect(e / 2 + 2, 0, e / 2 - 2, e)) : (this.context.beginPath(), this.context.translate(2, 0), this.context.moveTo(0, 0), this.context.lineTo(e - 2, e / 2), this.context.lineTo(0, e), this.context.fillStyle = "#fff", this.context.fill()), this.context.restore()
    }, nc.prototype.on = function(a, b) {
        this.canvas.addEventListener(a, b, !1)
    }, nc.prototype.off = function(a, b) {
        this.canvas.removeEventListener(a, b, !1)
    }, nc.prototype.destroy = function() {
        this.playing = !1, this.canvas.parentNode && this.container.removeChild(this.canvas)
    }, oc = {
        initialize: a,
        configure: m,
        sync: db,
        slide: cb,
        left: Mb,
        right: Nb,
        up: Ob,
        down: Pb,
        prev: Qb,
        next: Rb,
        navigateFragment: Fb,
        prevFragment: Hb,
        nextFragment: Gb,
        navigateTo: cb,
        navigateLeft: Mb,
        navigateRight: Nb,
        navigateUp: Ob,
        navigateDown: Pb,
        navigatePrev: Qb,
        navigateNext: Rb,
        layout: L,
        availableRoutes: pb,
        availableFragments: qb,
        toggleOverview: U,
        togglePause: $,
        toggleAutoSlide: ab,
        isOverview: V,
        isPaused: _,
        isAutoSliding: bb,
        addEventListeners: n,
        removeEventListeners: o,
        getState: Cb,
        setState: Db,
        getProgress: ub,
        getIndices: yb,
        getTotalSlides: zb,
        getSlide: Ab,
        getSlideBackground: Bb,
        getPreviousSlide: function() {
            return rc
        },
        getCurrentSlide: function() {
            return sc
        },
        getScale: function() {
            return Ec
        },
        getConfig: function() {
            return Ac
        },
        getQueryHash: function() {
            var a = {};
            location.search.replace(/[A-Z0-9]+?=([\w\.%-]*)/gi, function(b) {
                a[b.split("=").shift()] = b.split("=").pop()
            });
            for (var b in a) {
                var c = a[b];
                a[b] = r(unescape(c))
            }
            return a
        },
        isFirstSlide: function() {
            return 0 === pc && 0 === qc
        },
        isLastSlide: function() {
            return sc ? sc.nextElementSibling ? !1 : W(sc) && sc.parentNode.nextElementSibling ? !1 : !0 : !1
        },
        isReady: function() {
            return Bc
        },
        addEventListener: function(a, b, c) {
            "addEventListener" in window && (Gc.wrapper || document.querySelector(".reveal")).addEventListener(a, b, c)
        },
        removeEventListener: function(a, b, c) {
            "addEventListener" in window && (Gc.wrapper || document.querySelector(".reveal")).removeEventListener(a, b, c)
        },
        triggerKey: function(a) {
            Ub({
                keyCode: a
            })
        }
    }
});
