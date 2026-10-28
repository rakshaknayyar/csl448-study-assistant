import { useEffect } from "react";
import { ChevronLeft } from "lucide-react";

// A pop-up panel: centred on desktop, slides up from the bottom on phones.
// Closes with Escape, the Done button, or a click on the dark backdrop.
//
// One Sheet can show several screens (like iOS navigation): change `screen`
// to swap the content in place. The new content slides in from the right
// (or from the left when going back), and the backdrop never flickers
// because the Sheet itself stays open.
export default function Sheet({ title, subtitle, onClose, onBack, backLabel = "Back", screen, direction = "forward",
                                wide = false, children }) {
  useEffect(() => {
    const onKey = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="sheet-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className={`sheet${wide ? " wide" : ""}`} role="dialog" aria-modal="true" aria-label={title}>
        <div className={`sheet-head${onBack ? " with-back" : ""}`}>
          {onBack && (
            <button className="btn plain back" onClick={onBack}><ChevronLeft size={20} /> {backLabel}</button>
          )}
          <div className="titles">
            <h3>{title}</h3>
            {subtitle && <div className="sub">{subtitle}</div>}
          </div>
          <button className="btn plain" onClick={onClose}>Done</button>
        </div>
        <div className={`sheet-body${screen ? ` slide-${direction}` : ""}`} key={screen}>{children}</div>
      </div>
    </div>
  );
}
