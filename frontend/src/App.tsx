import React, { useState } from 'react'

function App() {
  const [query, setQuery] = useState('')

  const handleSearch = async () => {
    console.log('Searching for:', query)
  }

  return (
    <div className="app">
      <header>
        <h1>Market Research Platform</h1>
      </header>

      <main>
        <div className="search-box">
          <input
            type="text"
            placeholder="Enter topic to research..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button onClick={handleSearch}>Research</button>
        </div>
      </main>
    </div>
  )
}

export default App
