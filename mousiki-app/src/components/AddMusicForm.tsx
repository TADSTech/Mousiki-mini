import { useState } from 'react'
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
    <form onSubmit={handleSubmit} className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
      <h2 className="text-lg font-semibold text-white mb-4">Add a Track</h2>

      <div className="space-y-3 mb-4">
        <input
          type="text"
          placeholder="Track title"
          value={title}
          onChange={e => setTitle(e.target.value)}
          className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
        />
        <input
          type="text"
          placeholder="Artist name"
          value={artist}
          onChange={e => setArtist(e.target.value)}
          className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
        />
      </div>

      <p className="text-sm text-gray-400 mb-2">Genres:</p>
      <div className="flex flex-wrap gap-1.5 mb-4">
        {ALL_GENRES.map(g => (
          <button
            key={g}
            type="button"
            onClick={() => toggleGenre(g)}
            className={selectedGenres.includes(g) ? '' : 'opacity-50 hover:opacity-80'}
          >
            <GenreBadge genre={g} />
          </button>
        ))}
      </div>

      <button
        type="submit"
        disabled={!isValid}
        className={`w-full py-2 rounded-lg font-medium text-sm transition-all
          ${isValid
            ? 'bg-purple-600 hover:bg-purple-500 text-white cursor-pointer'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'}`}
      >
        + Add to Catalog
      </button>
    </form>
  )
}

export default AddMusicForm
