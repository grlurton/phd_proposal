"use strict";
var louchoff = louchoff || {};
louchoff.utils = {}, louchoff.external = {}, louchoff.Config = {
        fire: function(a, b, c) {
            var d = app.views;
            a = a.charAt(0).toUpperCase() + a.slice(1), b = void 0 === b ? "init" : b, "" !== a && d[a] && "function" == typeof d[a][b] && (console.log("fire :: " + a + "    || :: " + b + "(" + c + ")"), c = void 0 === c ? "" : c, d[a][b](c))
        },
        init: function() {
            if (Modernizr.addTest("hires", function() {
                    var a = window.devicePixelRatio || window.screen.deviceXDPI / window.screen.logicalXDPI || 1;
                    a > 1 && (document.body.className += " hires-screen")
                }), "Explorer" === louchoff.utils.BrowserDetect.browser && (document.documentElement.className += " ie"), "Firefox" === louchoff.utils.BrowserDetect.browser && (document.documentElement.className += " firefox"), "Chrome" === louchoff.utils.BrowserDetect.browser && (document.documentElement.className += " chrome"), "Safari" === louchoff.utils.BrowserDetect.browser && (document.documentElement.className += " safari"), document.body.className += navigator.userAgent.match(/(android|iphone|ipad|blackberry|symbian|symbianos|symbos|netfront|model-orange|javaplatform|iemobile|windows phone|samsung|htc|opera mobile|opera mobi|opera mini|presto|huawei|blazer|bolt|doris|fennec|gobrowser|iris|maemo browser|mib|cldc|minimo|semc-browser|skyfire|teashark|teleca|uzard|uzardweb|meego|nokia|bb10|playbook)/gi) ? screen.width >= 480 && screen.height >= 800 || screen.width >= 800 && screen.height >= 480 || navigator.userAgent.match(/ipad/gi) ? " tablette" : " mobile" : " desktop", "Explorer" === louchoff.utils.BrowserDetect.browser && parseInt(louchoff.utils.BrowserDetect.version) < 10) {
                var a = document.createElement("script");
                a.src = "scripts/libs/oldbrowsers/respond.min.js", document.getElementsByTagName("body").item(0).appendChild(a)
            }
            if ("Explorer" === louchoff.utils.BrowserDetect.browser && parseInt(louchoff.utils.BrowserDetect.version) < 9) {
                app.Settings.isOldIE = !0;
                var b = document.createElement("script");
                b.src = "scripts/libs/oldbrowsers/css3-mediaqueries.js", document.getElementsByTagName("body").item(0).appendChild(b)
            }
            console.log(louchoff.utils.BrowserDetect.browser + " - " + louchoff.utils.BrowserDetect.version + "  ||  PROPERTIES :: " + document.body.className), louchoff.Config.fire("Common");
            var c;
            $.each(document.body.className.replace(/-/g, "_").split(/\s+/), function(a, b) {
                c = b.split("_"), louchoff.Config.fire(c[0])
            }), app.Elem.body.trigger("page-complete")
        }
    },
    function(a, b) {
        var c = function(a, b, c) {
            var d;
            return function() {
                function e() {
                    c || a.apply(f, g), d = null
                }
                var f = this,
                    g = arguments;
                d ? clearTimeout(d) : c && a.apply(f, g), d = setTimeout(e, b || 100)
            }
        };
        jQuery.fn[b] = function(a) {
            return a ? this.bind("resize", c(a)) : this.trigger(b)
        }
    }(jQuery, "smartresize"), louchoff.Core = {
        init: function() {
            louchoff.Debug.init(), louchoff.utils.BrowserDetect.init(), louchoff.Config.init(), louchoff.external.Tracking.init()
        }
    }, louchoff.Display = {
        isOnView: function(a, b) {
            var c = !1;
            return a.offset().top - $(window).scrollTop() < $(window).height() - b && a.offset().top - $(window).scrollTop() > 0 && (c = !0), c
        }
    }, louchoff.Debug = {
        avoidConsole: function() {
            for (var a, b = function() {}, c = ["assert", "clear", "count", "debug", "dir", "dirxml", "error", "exception", "group", "groupCollapsed", "groupEnd", "info", "log", "markTimeline", "profile", "profileEnd", "table", "time", "timeEnd", "timeStamp", "trace", "warn"], d = c.length, e = window.console = window.console || {}; d--;) a = c[d], e[a] || (e[a] = b)
        },
        enablemobile: function() {
            $("body").append('<div id="console" style="font-size: 12px; color: #f00;    line-height: 10px;    margin-top:0px;    max-height: 100px;    overflow: auto;    z-index: 99999;    position:fixed;    top: 0px;    left:0px;    border:1px dotted #f00;    background-color: #fff;    opacity: 0.5;    width:80%;     display:block;  "></div>'), "undefined" != typeof console && (console.olog = "undefined" != typeof console.log ? console.log : function() {}), console.log = function(a) {
                console.olog(a), $("#console").append(a + "</br>")
            }, console.clear = function(a) {
                console.olog(a), $("#console").html("")
            }, console.error = console.debug = console.info = console.log
        },
        init: function() {
            louchoff.Debug.avoidConsole(), app.Settings.mobileDebug && (louchoff.Debug.enablemobile(), console.log("mobile debug actived"))
        }
    }, louchoff.external = {
        Tracking: {
            init: function() {
                this.addListeners()
            },
            addListeners: function() {
                $(document).on("click", ".tracking", function() {
                    try {
                        _gaq.push($(this).data("tracking").split(","))
                    } catch (a) {
                        console.log(a)
                    }
                })
            }
        }
    }, louchoff.utils.BrowserDetect = {
        browser: "",
        version: "",
        init: function() {
            this.browser = this.searchString(this.dataBrowser) || "Other", this.version = this.searchVersion(navigator.userAgent) || this.searchVersion(navigator.appVersion) || "Unknown"
        },
        searchString: function(a) {
            for (var b = 0; b < a.length; b++) {
                var c = a[b].string;
                if (this.versionSearchString = a[b].subString, -1 !== c.indexOf(a[b].subString)) return a[b].identity
            }
        },
        searchVersion: function(a) {
            var b = a.indexOf(this.versionSearchString);
            if (-1 !== b && "Trident" !== this.versionSearchString) return parseFloat(a.substring(b + this.versionSearchString.length + 1))
        },
        dataBrowser: [{
            string: navigator.userAgent,
            subString: "Chrome",
            identity: "Chrome"
        }, {
            string: navigator.userAgent,
            subString: "MSIE",
            identity: "Explorer"
        }, {
            string: navigator.userAgent,
            subString: "Trident",
            identity: "Explorer"
        }, {
            string: navigator.userAgent,
            subString: "Firefox",
            identity: "Firefox"
        }, {
            string: navigator.userAgent,
            subString: "Safari",
            identity: "Safari"
        }, {
            string: navigator.userAgent,
            subString: "Opera",
            identity: "Opera"
        }]
    };
var app = app || {};
app.views = app.views || {}, app.Settings = app.Settings || {}, app.Settings.debug = !1, app.Settings.mobileDebug = !1, app.Settings.environnment = "stage", app.Settings.isOldIE = !1, app.Globals = {}, app.Elem = {
    window: $(window),
    body: $("body"),
    document: $(document)
}, app.Mobile = app.Mobile || {}, app.Mobile.init = function() {
    console.log("mobile initialisation"), $(".reveal").addClass("no-reveal").removeClass("reveal"), $(".main-loader").fadeOut("slow"), app.Mobile.parseRevealElements()
}, app.Mobile.parseRevealElements = function() {
    $(".no-reveal").find('[data-src!=""]').each(function() {
        void 0 !== $(this).data("src") && $(this).attr("src", $(this).data("src"))
    }), $("#chapter-nav .reveal-link").each(function() {
        var a = $(this).attr("href").split("/");
        $(this).attr("href", "#" + a[1])
    })
}, app.Desktop = app.Desktop || {}, app.Desktop.modalVideoDisplayed = !1, app.Desktop.init = function() {
    console.log("desktop initialisation"), app.Desktop.isMobileDevice = /(iphone|ipod|ipad|android)/gi.test(navigator.userAgent), app.Desktop.initReveal(), app.Desktop.revealListeners(), app.Desktop.resize(), navigator.userAgent.match(/Android/i) && window.scrollTo(0, 1)
}, app.Desktop.resize = function() {
    if (!app.Desktop.isMobileDevice && /chrome/i.test(navigator.userAgent));
    else if (!app.Desktop.isMobileDevice && /firefox/i.test(navigator.userAgent));
    else {
        var a, b;
        b = Reveal.getScale(), a = .98 > b ? 40 * -(1 - b) + "%" : "20px", $(".legende").css({
            bottom: a
        })
    }
}, app.Desktop.checkVideoModal = function() {
    app.Desktop.modalVideoDisplayed === !1 ? -1 !== Reveal.getCurrentSlide().className.indexOf("has-video") && (app.views.Modal.update(window.modalVideoTexte), app.views.Modal.open(), app.Desktop.modalVideoDisplayed = !0) : app.views.Modal.close()
}, app.Desktop.initReveal = function() {
    Reveal.initialize({
        width: "100%",
	    height: "100%",
	    margin: 0,
	    minScale: 1,
	    maxScale: 1,
        keyboard: !0,
        fragments: 1,
        backgroundTransition: "slide",
        transition: "slide",
        transitionSpeed: "slow",
        controls: !1,
        progress: 1,
        history: !0,
        center: !0,
        mouseWheel: !0,
        theme: "custom",
        hideAddressBar: !0,
        help: !0,
        touch: !0,
        overview: !0,
        slideNumber: 1,
        viewDistance: 3,
        math: {
					mathjax: 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js',
					config: 'TeX-AMS_HTML-full'
				},
        dependencies: [
            {src: "reveal-js/js/plugin/zoom-js/zoom.js", async: !0} ,
            { src: 'plugin/math/math.js', async: true } ,
            { src: 'plugin/notes/notes.js', async: true }]
    })
}, app.Desktop.revealListeners = function() {
    $("#control-next").click(function() {}), $("#control-prev").click(function() {});
    var a = {};
    Reveal.addEventListener("slidechanged", function() {
        var b = Reveal.getCurrentSlide().id;
        a[b] = !0, app.Desktop.checkVideoModal()
    }), $(".reveal .slides").on("transitionend webkitTransitionEnd oTransitionEnd otransitionend MSTransitionEnd", function() {
        var b = Reveal.getCurrentSlide().id;
        a[b] && (a[b] = !1)
    }), Reveal.addEventListener("ready", function() {
        $(".main-loader").fadeOut("slow"), console.log("reveal ready"), app.Desktop.checkVideoModal()
    }), $(document).ready(function() {
        console.log("document ready")
    }), app.Elem.window.smartresize(function() {
        app.Desktop.resize()
    })
}, app.views.Modal = app.views.Modal || {}, app.views.Modal.$scope = $(".modal"), app.views.Modal.init = function() {
    app.views.Modal.addListenners()
}, app.views.Modal.addListenners = function() {
    app.views.Modal.$scope.find(".close-trigger").click(function() {
        app.views.Modal.close()
    })
}, app.views.Modal.open = function(a) {
    a === !0 && app.views.Modal.$scope.find(".btn-close").hide(), app.views.Modal.$scope.addClass("open")
}, app.views.Modal.close = function() {
    app.views.Modal.$scope.removeClass("open")
}, app.views.Modal.update = function(a) {
    app.views.Modal.$scope.find(".modal-text").html(a)
}, app.views.Home = app.views.Home || {}, app.views.Home.init = function() {
    app.views.Home.addListenners()
}, app.views.Home.addListenners = function() {
    $("body").hasClass("tablette") && window.addEventListener("orientationchange", function() {
        app.views.Home.onOrientationChange()
    }, !1)
}, app.views.Home.onOrientationChange = function() {
    switch (console.log("orientation :: " + window.orientation), window.orientation) {
        case 0:
            app.views.Modal.update(window.modalOrientationTexte), app.views.Modal.open(!0);
            break;
        case 90:
            app.views.Modal.close();
            break;
        case -90:
            app.views.Modal.close();
            break;
        case 180:
            app.views.Modal.update(window.modalOrientationTexte), app.views.Modal.open(!0)
    }
}, app.views.Nav = app.views.Nav || {}, app.views.Nav.$scope = $("#chapter-nav"), app.views.Nav.init = function() {
    app.views.Nav.addListenners()
}, app.views.Nav.addListenners = function() {
    $("#btn-menu").on("mouseenter focus", function() {
        $(this).addClass("in")
    }).on("mouseleave blur", function() {
        $(this).removeClass("in")
    }).click(function() {
        $(this).hasClass("opened") ? ($(this).removeClass("opened"), app.views.Nav.$scope.removeClass("opened")) : ($(this).addClass("opened"), app.views.Nav.$scope.addClass("opened")), $(this).blur()
    }), app.views.Nav.$scope.find("a").click(function() {
        $("#btn-menu").trigger("click")
    })
}, app.views.Nav.open = function() {
    app.views.Nav.$scope.css("right", "0px")
}, app.views.Nav.close = function() {
    app.views.Nav.$scope.css("right", -app.views.Nav.$scope.width() + "px")
}, $(document).ready(function() {
    louchoff.Core.init(), app.views.Home.init(), app.views.Modal.init(), app.views.Nav.init(), $("body").hasClass("mobile") ? (console.log("mobile"), app.Mobile.init()) : (console.log("tablette // desktop"), $("body").hasClass("tablette") && app.views.Home.onOrientationChange(), app.Desktop.init())
});
