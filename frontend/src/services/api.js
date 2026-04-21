const API_BASE = '/api'

export async function uploadVideo(file) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
    })

    if (!response.ok) {
        const err = await response.text()
        throw new Error(`Upload failed: ${err}`)
    }

    return response.json()
}

export async function getJobStatus(jobId) {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`)

    if (!response.ok) {
        throw new Error(`Failed to fetch job status: ${response.statusText}`)
    }

    return response.json()
}

export async function loginUser(email, password) {
    const formData = new URLSearchParams()
    formData.append('username', email)
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

export async function registerUser(name, email, password, role) {
    const response = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ full_name: name, email, password, role: role.toUpperCase() })
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Registration failed')
    }
    return response.json()
}
