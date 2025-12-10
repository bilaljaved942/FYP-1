import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const uploadVideo = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const getJobStatus = async (jobId) => {
  const response = await api.get(`/jobs/${jobId}`)
  return response.data
}
