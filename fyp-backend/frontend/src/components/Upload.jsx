import { useState, useRef } from 'react'
import { Upload as UploadIcon, FileVideo, X } from 'lucide-react'
import { cn } from '../lib/utils'

export default function Upload({ onUpload, isUploading }) {
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = () => {
    if (file) {
      onUpload(file)
    }
  }

  const removeFile = () => {
    setFile(null)
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={cn(
          'relative border-2 border-dashed rounded-xl p-12 transition-all duration-200 ease-in-out',
          dragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400 bg-white',
          isUploading && 'opacity-50 pointer-events-none'
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isUploading}
        />
        
        <div className="flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 mb-4 rounded-full bg-blue-100 flex items-center justify-center">
            <UploadIcon className="w-8 h-8 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Upload Classroom Video
          </h3>
          <p className="text-sm text-gray-500 mb-4">
            Drag and drop your video file here, or click to browse
          </p>
          <p className="text-xs text-gray-400">
            Supported formats: MP4, AVI, MOV, MKV
          </p>
        </div>
      </div>

      {file && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileVideo className="w-8 h-8 text-blue-600" />
            <div>
              <p className="text-sm font-medium text-gray-900">{file.name}</p>
              <p className="text-xs text-gray-500">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          </div>
          <button
            onClick={removeFile}
            className="p-1 hover:bg-gray-200 rounded-full transition-colors"
            disabled={isUploading}
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={!file || isUploading}
        className={cn(
          'mt-6 w-full py-3 px-6 rounded-lg font-semibold text-white transition-all duration-200',
          file && !isUploading
            ? 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
            : 'bg-gray-300 cursor-not-allowed'
        )}
      >
        {isUploading ? 'Uploading...' : 'Analyze Video'}
      </button>
    </div>
  )
}
