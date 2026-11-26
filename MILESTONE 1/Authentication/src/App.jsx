import { useState } from 'react';
import {
  Activity, ArrowRight, Brain, Check, CheckCircle2, Crosshair, Eye, EyeOff,
  HeartPulse, LockKeyhole, Mail, ShieldCheck, Shield, UserRound, UserRoundPlus
} from 'lucide-react';

function HeartArtwork() {
  return (
    <div className="heart-art" aria-label="Illustration of a human heart and coronary arteries" role="img">
      <div className="heart-orbit orbit-one" /><div className="heart-orbit orbit-two" />
      <svg viewBox="0 0 360 420" className="heart-svg" aria-hidden="true">
        <defs>
          <linearGradient id="heartFill" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#c7edff"/><stop offset=".48" stopColor="#5ca9ed"/><stop offset="1" stopColor="#1761b5"/></linearGradient>
          <linearGradient id="arteryFill" x1="0" y1="0" x2="0.8" y2="1"><stop stopColor="#b9e9ff"/><stop offset="1" stopColor="#2f80cb"/></linearGradient>
          <filter id="heartGlow"><feGaussianBlur stdDeviation="7" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <g filter="url(#heartGlow)" opacity=".95" stroke="#d9f4ff" strokeWidth="2">
          <path d="M135 125 C128 84 126 45 145 34 C163 24 176 48 173 77 L179 111" fill="url(#arteryFill)"/>
          <path d="M171 110 C173 71 190 22 210 23 C230 24 227 60 213 88 L201 127" fill="url(#arteryFill)"/>
          <path d="M197 115 C211 80 237 49 253 59 C271 70 245 101 233 126" fill="url(#arteryFill)"/>
          <path d="M151 124 C136 93 103 84 91 103 C78 123 109 148 126 161" fill="url(#arteryFill)"/>
          <path d="M187 117 C218 89 261 100 276 126 C290 150 263 171 244 184" fill="url(#arteryFill)"/>
          <path d="M130 139 C91 147 65 187 77 231 C89 278 120 318 160 350 C183 369 204 357 220 338 C250 301 275 254 264 211 C257 182 236 162 210 151 C189 140 161 128 130 139Z" fill="url(#heartFill)" stroke="#d9f4ff" strokeWidth="3"/>
        </g>
        <g fill="none" strokeLinecap="round" strokeLinejoin="round">
          <path d="M143 155 C154 181 166 202 165 231 C165 267 184 291 200 321" stroke="#ffb8a2" strokeWidth="9"/>
          <path d="M143 155 C154 181 166 202 165 231 C165 267 184 291 200 321" stroke="#ff684f" strokeWidth="4"/>
          <path d="M165 205 C137 210 121 235 116 261 M165 230 C190 216 215 222 231 243 M182 270 C157 274 145 295 145 311 M201 310 C224 292 234 276 238 257" stroke="#ffb9a5" strokeWidth="7"/>
          <path d="M165 205 C137 210 121 235 116 261 M165 230 C190 216 215 222 231 243 M182 270 C157 274 145 295 145 311 M201 310 C224 292 234 276 238 257" stroke="#ff6c57" strokeWidth="3"/>
          <path d="M124 171 C109 187 105 206 109 224 M210 160 C231 172 243 191 245 211" stroke="#f6ffff" strokeWidth="3" opacity=".85"/>
        </g>
        <path d="M109 176 C132 155 153 154 170 169 C185 182 190 204 184 226 C179 251 189 277 207 302" fill="none" stroke="#eafaff" strokeWidth="2" opacity=".65"/>
      </svg>
      <div className="stenosis-callout"><span className="callout-vessel"><i /></span><span className="callout-label">AI stenosis insight</span></div>
      <div className="art-caption"><span className="caption-dot"/> Coronary artery visualization</div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState('login');
  const [role, setRole] = useState('Researcher');
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');

  const switchMode = (next) => { setMode(next); setNotice(''); setError(''); };
  const submit = (event) => {
    event.preventDefault(); setError(''); setNotice('');
    if (mode === 'signup' && !name.trim()) { setError('Please enter your full name.'); return; }
    if (!/^\S+@\S+\.\S+$/.test(email)) { setError('Enter a valid email address.'); return; }
    if (password.length < 8) { setError('Password must contain at least 8 characters.'); return; }
    setNotice(mode === 'login'
      ? `Form validated for ${role.toLowerCase()} access. Connect your backend to complete sign-in.`
      : `Sign-up details validated for ${role.toLowerCase()}. Connect your backend to create the account.`);
  };

  return (
    <main className="page-shell">
      <div className="ambient ambient-top"/><div className="ambient ambient-bottom"/>
      <section className="intro-panel">
        <a className="brand" href="#top" aria-label="Artery AI home">
          <span className="brand-icon"><HeartPulse size={39} strokeWidth={1.8}/></span>
          <span><strong>Artery AI</strong><small>Smarter Analysis. Healthier Hearts.</small></span>
        </a>
        <div className="intro-copy">
          <div className="eyebrow"><span/> INTELLIGENT CARDIOVASCULAR IMAGING</div>
          <h1>See beyond the scan.<br/><span>Discover what matters.</span></h1>
          <p className="intro-description">AI-powered artery segmentation and stenosis detection for clearer insights and more confident research.</p>
          <div className="feature-list">
            <div className="feature"><span className="feature-icon"><Crosshair size={22}/></span><div><strong>Precision Detection</strong><p>Identify arterial narrowing with AI-assisted analysis.</p></div></div>
            <div className="feature"><span className="feature-icon"><Brain size={22}/></span><div><strong>Intelligent Analysis</strong><p>Deep learning insights, designed for researchers.</p></div></div>
            <div className="feature"><span className="feature-icon"><ShieldCheck size={22}/></span><div><strong>Secure Workspace</strong><p>Role-based access for your research workflow.</p></div></div>
          </div>
        </div>
        <HeartArtwork />
        <div className="intro-footer"><Activity size={28}/><span>Better Diagnostics</span><i/> <span>Better Decisions</span><i/> <span>Better Outcomes</span></div>
      </section>

      <section className="auth-side" aria-label="Account access">
        <div className="auth-card">
          <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
            <button type="button" role="tab" aria-selected={mode === 'login'} className={mode === 'login' ? 'tab active' : 'tab'} onClick={() => switchMode('login')}><UserRound size={19}/> Login</button>
            <button type="button" role="tab" aria-selected={mode === 'signup'} className={mode === 'signup' ? 'tab active' : 'tab'} onClick={() => switchMode('signup')}><UserRoundPlus size={19}/> Sign Up</button>
          </div>
          <div className="form-content">
            <div className="form-heading"><span className="secure-pill"><Shield size={14}/> SECURE ACCESS</span><h2>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h2><p>{mode === 'login' ? 'Sign in to continue to your workspace.' : 'Join your team’s intelligent imaging workspace.'}</p></div>
            <form onSubmit={submit} noValidate>
              {mode === 'signup' && <label className="field-label">Full name<div className="input-wrap"><UserRound size={18}/><input value={name} onChange={e => setName(e.target.value)} placeholder="Enter your full name" autoComplete="name"/></div></label>}
              <fieldset className="role-field"><legend>I am signing in as</legend><div className="role-options">
                <button type="button" className={`role-option ${role === 'Researcher' ? 'selected' : ''}`} onClick={() => setRole('Researcher')}><UserRound size={19}/><span>Researcher</span>{role === 'Researcher' && <CheckCircle2 className="role-check" size={18}/>}</button>
                <button type="button" className={`role-option ${role === 'Administrator' ? 'selected' : ''}`} onClick={() => setRole('Administrator')}><ShieldCheck size={19}/><span>Administrator</span>{role === 'Administrator' && <CheckCircle2 className="role-check" size={18}/>}</button>
              </div></fieldset>
              <label className="field-label">Email address<div className="input-wrap"><Mail size={18}/><input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@organization.com" autoComplete="email"/></div></label>
              <label className="field-label">Password<div className="input-wrap"><LockKeyhole size={18}/><input type={showPassword ? 'text' : 'password'} value={password} onChange={e => setPassword(e.target.value)} placeholder={mode === 'signup' ? 'Create at least 8 characters' : 'Enter your password'} autoComplete={mode === 'login' ? 'current-password' : 'new-password'}/><button className="icon-button" type="button" onClick={() => setShowPassword(!showPassword)} aria-label={showPassword ? 'Hide password' : 'Show password'}>{showPassword ? <EyeOff size={18}/> : <Eye size={18}/>}</button></div></label>
              {mode === 'login' && <div className="form-options"><label className="remember"><input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)}/><span className="fake-check"><Check size={12}/></span>Remember me</label><button type="button" className="text-button" onClick={() => {setNotice('Password recovery is ready to connect to your backend or email service.'); setError('');}}>Forgot password?</button></div>}
              {error && <div className="message error" role="alert">{error}</div>}
              {notice && <div className="message success" role="status">{notice}</div>}
              <button className="submit-button" type="submit">{mode === 'login' ? 'Sign in to workspace' : 'Create account'}<ArrowRight size={18}/></button>
            </form>
            <div className="divider"><span/> <small>OR CONTINUE WITH</small> <span/></div>
            <button className="google-button" type="button" onClick={() => {setNotice('Google sign-in needs OAuth credentials and backend integration.'); setError('');}}><span className="google-g">G</span>Continue with Google</button>
            <p className="switch-prompt">{mode === 'login' ? 'New to Artery AI?' : 'Already have an account?'} <button type="button" onClick={() => switchMode(mode === 'login' ? 'signup' : 'login')}>{mode === 'login' ? 'Create account' : 'Sign in'}</button></p>
            <p className="privacy-note"><ShieldCheck size={14}/> Your workspace is designed with privacy in mind.</p>
          </div>
        </div>
        <p className="copyright">© 2026 Artery AI <span>•</span> Research &amp; Clinical Innovation</p>
      </section>
    </main>
  );
}
