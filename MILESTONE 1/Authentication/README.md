# Artery AI — React Login / Sign Up UI

Responsive React + Vite authentication screen for the Artery AI artery segmentation and stenosis detection project.

## Run locally

1. Install Node.js (LTS) if it is not already installed.
2. Open a terminal in this folder.
3. Run:

   ```bash
   npm install
   npm run dev
   ```

4. Open the local URL printed by Vite (usually `http://localhost:5173`).

## Included

- Responsive medical-themed layout
- Login and Sign Up tabs
- Researcher / Administrator role selection
- Email and password fields with validation
- Show / hide password
- Remember-me checkbox
- Forgot-password and Google sign-in integration placeholders
- Accessible labels and status messages

## Backend integration note

This is the frontend UI. The form currently validates inputs in the browser and displays a message; it does not authenticate users or persist accounts. Connect `submit` in `src/App.jsx` to your backend login/register endpoints, and configure Google OAuth before using those options in production. Role selection in the frontend is not an authorization boundary—enforce roles and permissions on the server.
