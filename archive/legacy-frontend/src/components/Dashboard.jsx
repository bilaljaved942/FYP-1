import { useState, useEffect, useCallback } from 'react'
import { Loader2, CheckCircle, AlertCircle, RotateCcw } from 'lucide-react'
import Upload from './Upload'
import {
  EmotionChart,
  ActionsChart,
  EngagementSummaryChart,
  EngagementTimelineChart,
} from './Charts'
import { uploadVideo, getJobStatus } from '../api/client'

const POLL_INTERVAL = 2000 // 2 seconds

export default function Dashboard() {
  const [state, setState] = useState('upload') // 'upload' | 'processing' | 'completed' | 'error'
  const [jobId, setJobId] = useState(null)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [isUploading, setIsUploading] = useState(false)

  const pollJobStatus = useCallback(async (id) => {
    try {
      const data = await getJobStatus(id)
      
      if (data.status === 'COMPLETED') {
        setResults(data.ai_results)
        setState('completed')
        return true // Stop polling
      } else if (data.status === 'FAILED') {
        setError('Video processing failed. Please try again.')
        setState('error')
        return true // Stop polling
      }
      
      return false // Continue polling
    } catch (err) {
      console.error('Polling error:', err)
      setError('Failed to check job status. Please try again.')
      setState('error')
      return true // Stop polling on error
    }
  }, [])

  useEffect(() => {
    if (state !== 'processing' || !jobId) return

    let isMounted = true
    let timeoutId = null

    const poll = async () => {
      if (!isMounted) return
      
      const shouldStop = await pollJobStatus(jobId)
      
      if (!shouldStop && isMounted) {
        timeoutId = setTimeout(poll, POLL_INTERVAL)
      }
    }

    poll()

    return () => {
      isMounted = false
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [state, jobId, pollJobStatus])

  const handleUpload = async (file) => {
    setIsUploading(true)
    setError(null)

    try {
      const response = await uploadVideo(file)
      setJobId(response.job_id)
      setState('processing')
    } catch (err) {
      console.error('Upload error:', err)
      setError('Failed to upload video. Please try again.')
      setState('error')
    } finally {
      setIsUploading(false)
    }
  }

  const handleReset = () => {
    setState('upload')
    setJobId(null)
    setResults(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Student Engagement Dashboard
              </h1>
              <p className="mt-1 text-sm text-gray-500">
                Upload classroom videos to analyze student engagement
              </p>
            </div>
            {state !== 'upload' && (
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                New Analysis
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Upload State */}
        {state === 'upload' && (
          <div className="flex items-center justify-center min-h-[60vh]">
            <Upload onUpload={handleUpload} isUploading={isUploading} />
          </div>
        )}

        {/* Processing State */}
        {state === 'processing' && (
          <div className="flex flex-col items-center justify-center min-h-[60vh]">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
              <Loader2 className="w-16 h-16 text-blue-600 animate-spin mx-auto mb-6" />
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                Analyzing Video...
              </h2>
              <p className="text-gray-500 mb-4">
                Our AI is processing your classroom video
              </p>
              <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
                <span className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
                <span>This may take a few moments</span>
              </div>
            </div>
          </div>
        )}

        {/* Error State */}
        {state === 'error' && (
          <div className="flex flex-col items-center justify-center min-h-[60vh]">
            <div className="bg-white rounded-xl shadow-sm border border-red-200 p-12 text-center">
              <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-6" />
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                Something went wrong
              </h2>
              <p className="text-gray-500 mb-6">{error}</p>
              <button
                onClick={handleReset}
                className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {/* Results State */}
        {state === 'completed' && results && (
          <div className="space-y-6">
            {/* Success Banner */}
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-green-800 font-medium">
                Analysis complete! Here are your results.
              </span>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <EmotionChart data={results.emotions} />
              <ActionsChart data={results.actions} />
              <EngagementSummaryChart data={results.actions} />
              <EngagementTimelineChart data={results.engagement_over_time} />
            </div>

            {/* Raw Data */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Raw Analysis Data
              </h3>
              <pre className="bg-gray-50 rounded-lg p-4 overflow-x-auto text-sm text-gray-700">
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
