import { ShieldCheck, Home, ChartNoAxesColumn, CircleHelp, Sun, Moon } from 'lucide-react';

export default function Topbar({ page, onNavigate, theme, onToggleTheme }) {
  return <header className="topbar">
    <button className="brand" onClick={() => onNavigate('home')} aria-label="Checkie home">
      <span className="brand-mark"><ShieldCheck size={19} /></span>checkie
    </button>
    <nav aria-label="Main navigation">
      <button className={page === 'home' ? 'nav active' : 'nav'} onClick={() => onNavigate('home')}><Home size={15} /> Home</button>
      <button className={page === 'dashboard' ? 'nav active' : 'nav'} onClick={() => onNavigate('dashboard')}><ChartNoAxesColumn size={15} /> Dashboard</button>
      <button className={page === 'about' ? 'nav active' : 'nav'} onClick={() => onNavigate('about')}><CircleHelp size={15} /> How it works</button>
    </nav>
    <button className="theme-toggle" type="button" onClick={onToggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
      {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />} <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
    </button>
  </header>;
}
