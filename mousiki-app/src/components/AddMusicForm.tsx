import { useState } from 'react'
import { Plus, Music, User } from 'lucide-react'
import type { Genre, Track } from '../engine/types'
import { ALL_GENRES, generateId } from '../engine/catalog'
import GenreBadge from './GenreBadge'

interface AddMusicFormProps {
  onAdd: (track: Track) => void
}

const AddMusicForm = ({ onAdd }: AddMusicFormProps) => {
  const [title, setTitle] = useState('')
  const [artist, setArtist] = useState('')
  const [selectedGenres, setSelectedGenres] = useState<Genre[]>([])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || !artist.trim() || selectedGenres.length === 0) return
    onAdd({
      id: generateId(),
      title: title.trim(),
      artist: artist.trim(),
      genres: selectedGenres,
    })
    setTitle('')
    setArtist('')
    setSelectedGenres([])
  }

  const toggleGenre = (g: Genre) => {
    setSelectedGenres(prev =>
      prev.includes(g) ? prev.filter(x => x !== g) : [...prev, g]
    )
  }

  const isValid = title.trim() && artist.trim() && selectedGenres.length > 0

  return (
    <form onSubmit={handleSubmit} className="glass-panel rounded-3xl p-6 md:p-8 sticky top-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2.5 bg-accent-500/10 rounded-xl text-accent-400">
          <Plus className="w-5 h-5" />
        </div>
        <h2 className="text-xl font-bold text-gray-50">Add Track</h2>
      </div>

      <div className="space-y-4 mb-6">
        <div className="relative group">
          <Music className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 group-focus-within:text-accent-400 transition-colors" />
          <input
            type="text"
            placeholder="Track title"
            value={title}
            onChange={e => setTitle(e.target.value)}
            className="w-full pl-12 pr-4 py-3.5 bg-gray-900/50 border border-gray-700/50 rounded-2xl text-gray-50 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent-500/50 focus:border-accent-500/50 transition-all shadow-inner"
          />
        </div>
        <div className="relative group">
          <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 group-focus-within:text-accent-400 transition-colors" />
          <input
            type="text"
            placeholder="Artist name"
            value={artist}
            onChange={e => setArtist(e.target.value)}
            className="w-full pl-12 pr-4 py-3.5 bg-gray-900/50 border border-gray-700/50 rounded-2xl text-gray-50 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent-500/50 focus:border-accent-500/50 transition-all shadow-inner"
          />
        </div>
      </div>

      <div className="mb-8">
        <p className="text-sm font-medium text-gray-400 mb-3 ml-1">Select Genres:</p>
        <div className="flex flex-wrap gap-2">
          {ALL_GENRES.map(g => (
            <button
              key={g}
              type="button"
              onClick={() => toggleGenre(g)}
              className={`transition-all duration-200 ${
                selectedGenres.includes(g) 
                  ? 'scale-105 shadow-lg shadow-brand-500/20' 
                  : 'opacity-40 hover:opacity-80 grayscale hover:grayscale-0'
              }`}
            >
              <GenreBadge genre={g} />
            </button>
          ))}
        </div>
      </div>

      <button
        type="submit"
        disabled={!isValid}
        className={`w-full py-4 rounded-2xl font-bold text-sm tracking-wide uppercase transition-all duration-300 flex items-center justify-center gap-2
          ${isValid
            ? 'bg-gradient-to-r from-accent-600 to-brand-600 hover:from-accent-500 hover:to-brand-500 text-white shadow-lg shadow-accent-500/25 cursor-pointer transform hover:-translate-y-0.5'
            : 'bg-gray-800 text-gray-500 cursor-not-allowed opacity-50'}`}
      >
        <Plus className="w-5 h-5" />
        Add to Catalog
      </button>
    </form>
  )
}

export default AddMusicForm
