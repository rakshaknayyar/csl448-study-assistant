// Small iOS-style controls used in the settings sheet.

export function Switch({ checked, onChange, label }) {
  return (
    <label className="switch">
      <input type="checkbox" role="switch" aria-label={label} checked={checked}
             onChange={(event) => onChange(event.target.checked)} />
      <span className="track"><span className="knob" /></span>
    </label>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button key={option.value} role="radio" aria-checked={value === option.value}
                className={value === option.value ? "on" : ""} onClick={() => onChange(option.value)}>
          {option.label}
        </button>
      ))}
    </div>
  );
}
