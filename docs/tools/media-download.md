# Media Download Tool

The model can use the media download tool to search for media content requested
by the user and download a selected result into the sandboxed workspace.

## Tool names

Common aliases include:

- `download_media`
- `search_and_download_media`
- `download_media_content`
- `search_download_media`
- `media_download`

## Parameters

- `query` (or `search_query`, `media_query`, `request`, `keyword`, `keywords`):
  media search query.
- `media_type` (or `type`, `kind`, `content_type`): `image`, `video`, or `audio`.
- `url`: direct media URL to download without searching.
- `save_path` (or `destination`, `output_path`, `filename`, `path`): destination
  path inside the workspace.

## Behavior

1. Checks network availability.
2. Searches for media when no direct URL is supplied.
3. Extracts a direct media URL from search results, including DuckDuckGo
   redirect URLs.
4. Downloads the selected media into the workspace using `download_web_file`.
5. Returns concise status text and the saved file path.

## Tests

- `tests/unit/test_media_download.py`
