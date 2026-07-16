import { useState } from 'react'
import type { Track, Genre } from '../engine/types'
import { ALL_GENRES } from '../engine/catalog'
import TrackCard from './TrackCard'

interface MusicLibraryProps {
  tracks: Track[]
  selectedTrack: Track | null
  onSelectTrack: (track: Track) => void
  onRemoveTrack: (id: string) => void
}

const MusicLibrary = ({ tracks, selectedTrack, onSelectTrack, onRemoveTrack }: MusicLibraryProps) => {
  const [search, setSearch] = useState('')
  const [genreFilter, setGenreFilter] = useState<Genre | 'All'>('All')

  const filtered = tracks.filter(t => {
    if (genreFilter !== 'All' && !t.genres.includes(genreFilter)) return false
    if (search) {
      const q = search.toLowerCase()
      if (!t.title.toLowerCase().includes(q) && !t.artist.toLowerCase().includes(q)) return false
    }
    return true
  })

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-4">
        <input
          type="text"
          placeholder="Search tracks or artists..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="flex-1 min-w-[200px] px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
        />
        <select
          value={genreFilter}
          onChange={e => setGenreFilter(e.target.value as Genre | 'All')}
          className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <option value="All">All Genres</option>
          {ALL_GENRES.map(g => <option key={g} value={g}>{g}</option>)}
        </select>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map(track => (
          <TrackCard
            key={track.id}
            track={track}
            selected={selectedTrack?.id === track.id}
            onSelect={onSelectTrack}
            onRemove={onRemoveTrack}
          />
        ))}
      </div>

      {filtered.length === 0 && (
        <p className="text-gray-500 text-center py-8">No tracks match your filters. Add some music!</p>
      )}
    </div>
  )
}

export default MusicLibrary
