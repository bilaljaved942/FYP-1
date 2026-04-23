const API_BASE = '/api'

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

export async function registerUser(name, email, password, role, university, department) {
    const response = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            full_name: name, 
            email, 
            password, 
            role: role.toUpperCase(),
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
