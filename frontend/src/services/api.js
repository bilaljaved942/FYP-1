/**
 * API Service Layer
 * ==================
 * Centralized module for all HTTP requests to the FastAPI backend.
 *
 * All functions communicate with the backend via the `/api` prefix,
 * which is proxied by Vite's dev server to avoid CORS issues.
 *
 * Functions:
 *   uploadVideo()      → POST /upload (multipart form with video + metadata)
 *   getJobStatus()     → GET  /jobs/{jobId} (poll for analysis results)
 *   loginUser()        → POST /auth/login (OAuth2 password flow)
 *   registerUser()     → POST /auth/register (JSON body)
 *   getHodAnalytics()  → GET  /analytics/hod (department aggregated data)
 */

// Base URL for all API calls — proxied by vite.config.js in development
const API_BASE = '/api'


/**
 * Upload a classroom video for AI analysis.
 *
 * @param {File}   file         - The MP4 video file to analyze.
 * @param {string} classSection - Class section identifier (e.g., "CS-8A").
 * @param {string} courseName   - Course name (e.g., "GenAI").
 * @param {string} token        - JWT authentication token.
 * @returns {Promise<{job_id: string, message: string}>} - Upload response with job ID.
 */
export async function uploadVideo(file, classSection, courseName, token) {
    const formData = new FormData()
    formData.append('file', file)
    if (classSection) formData.append('class_section', classSection)
    if (courseName) formData.append('course_name', courseName)

    const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`
        },
        body: formData,
    })

    if (!response.ok) {
        const err = await response.text()
        throw new Error(`Upload failed: ${err}`)
    }

    return response.json()
}


/**
 * Poll the status of a video analysis job.
 * Called every 3 seconds by the frontend until status is COMPLETED or FAILED.
 *
 * @param {string} jobId - UUID of the analysis job.
 * @returns {Promise<{status: string, ai_results: object|null}>} - Job status and results.
 */
export async function getJobStatus(jobId) {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`)

    if (!response.ok) {
        throw new Error(`Failed to fetch job status: ${response.statusText}`)
    }

    return response.json()
}


/**
 * Authenticate a user with email and password.
 * Uses OAuth2 password flow (application/x-www-form-urlencoded).
 *
 * @param {string} email    - User's email address.
 * @param {string} password - User's password.
 * @returns {Promise<{access_token: string, role: string, full_name: string}>}
 */
export async function loginUser(email, password) {
    // OAuth2 password flow requires URL-encoded form data
    const formData = new URLSearchParams()
    formData.append('username', email)   // OAuth2 spec uses 'username' field
    formData.append('password', password)

    const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Login failed')
    }
    return response.json()
}


/**
 * Register a new Teacher or HOD account.
 * Sends a JSON body with user details and institutional metadata.
 *
 * @param {string} name       - Full name (displayed in navbar).
 * @param {string} email      - Email address (used for login).
 * @param {string} password   - Password (hashed server-side with bcrypt).
 * @param {string} role       - "teacher" or "hod" (converted to uppercase for backend).
 * @param {string} university - University name (e.g., "FAST NUCES").
 * @param {string} department - Department name (e.g., "Computer Science").
 * @returns {Promise<{access_token: string, role: string}>}
 */
export async function registerUser(name, email, password, role, university, department) {
    const response = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            full_name: name, 
            email, 
            password, 
            role: role.toUpperCase(),  // Backend expects "TEACHER" or "HOD"
            university,
            department
        })
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Registration failed')
    }
    return response.json()
}


/**
 * Fetch aggregated analytics for the HOD dashboard.
 * Returns department-wide metrics including teacher comparisons,
 * emotion/action distributions, and course-level breakdowns.
 *
 * @param {string} token - JWT authentication token (must be an HOD user).
 * @returns {Promise<object>} - Analytics data including KPIs, charts, and breakdowns.
 */
export async function getHodAnalytics(token) {
    const response = await fetch(`${API_BASE}/analytics/hod`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })

    if (!response.ok) {
        throw new Error('Failed to fetch analytics')
    }

    return response.json()
}
