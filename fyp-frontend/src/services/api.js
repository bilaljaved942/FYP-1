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
