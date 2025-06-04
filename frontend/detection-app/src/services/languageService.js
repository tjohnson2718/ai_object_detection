const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const languageService = {
    async parseQuery(query) {
        try {
            const response = await fetch(`${API_URL}/language/parse_query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            })

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}, Details: ${response.statusText}`)
            }

            return await response.json()
        } catch (error) {
            console.error('Error parsing query:', error)
            throw error
        }
    }
}