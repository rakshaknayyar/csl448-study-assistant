import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft } from "lucide-react";

// How long the closing animation runs (keep in step with .closing in styles.css).
const CLOSE_MS = 220;

// A pop-up panel: centred on desktop, slides up from the bottom on phones.
// Closes with Escape, the Done button, or a click on the dark backdrop -
// it first plays a short exit animation, then tells the parent to remove it.
//
// One Sheet can show several screens (like iOS navigation): change `screen`
// to swap the content in place. The new content slides in from the right
// (or from the left when going back), and the backdrop never flickers
// because the Sheet itself stays open.
export default function Sheet({ title, subtitle, onClose, onBack, backLabel = "Back", screen, direction = "forward",
                                wide = false, children }) {
  const [closing, setClosing] = useState(false);
  const timer = useRef(null);

  const close = useCallback(() => {
    if (timer.current) return;                         // already closing
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { onClose(); return; }
    setClosing(true);
    timer.current = setTimeout(onClose, CLOSE_MS);
  }, [onClose]);

  useEffect(() => () => clearTimeout(timer.current), []);

  useEffect(() => {
    const onKey = (event) => event.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  return (
    <div className={`sheet-backdrop${closing ? " closing" : ""}`}
         onMouseDown={(event) => event.target === event.currentTarget && close()}>
      <div className={`sheet${wide ? " wide" : ""}`} role="dialog" aria-modal="true" aria-label={title}>
        <div className={`sheet-head${onBack ? " with-back" : ""}`}>
          {onBack && (
            <button className="btn plain back" onClick={onBack}><ChevronLeft size={20} /> {backLabel}</button>
          )}
          <div className="titles">
            <h3>{title}</h3>
            {subtitle && <div className="sub">{subtitle}</div>}
          </div>
          <button className="btn plain" onClick={close}>Done</button>
        </div>
        <div className={`sheet-body${screen ? ` slide-${direction}` : ""}`} key={screen}>{children}</div>
      </div>
    </div>
  );
}
