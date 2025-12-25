# ADR 001: API Versioning Strategy

## Status
Accepted

## Date
2025-12-11

## Context
The Nemo Server API Gateway serves as the single entry point for all microservices. As the platform moves towards enterprise readiness, we expect breaking changes in API contracts. A versioning strategy is required to:
1.  Allow safe evolution of APIs without breaking existing clients (Mobile App, Frontend).
2.  Provide clear deprecation paths.
3.  Support multiple active API versions if necessary.

## Decision
We have decided to adopt **URL Path Versioning** (e.g., `/api/v1/resource`).

### Rationale
-   **Explicitness**: The version is clearly visible in the URI, making debugging and logging easier.
-   **Routing**: Layer 7 load balancers (Nginx) and the API Gateway can easily route based on path prefixes.
-   **Tooling Support**: Swagger/OpenAPI generators handle path versioning natively.

### Alternatives Considered
-   **Header Versioning** (`Accept: application/vnd.nemo.v1+json`): Considered too complex for simple clients and harder to test via browser/curl.
-   **Query Parameter** (`?version=1`): Less RESTful and harder to cache.

## Implementation
-   The API Gateway (`services/api-gateway`) will mount a `v1` `APIRouter` at `/api/v1`.
-   Existing "legacy" routes (e.g., `/api/auth/login`) will be maintained as aliases to the v1 handlers to ensure backward compatibility for the immediate future.
-   New endpoints will **only** be registered under `/api/v1`.

## Consequences
-   **Positive**: Clear separation of concerns. New features can break contracts safely in v2.
-   **Negative**: Slight code duplication if handlers are not properly shared (mitigated by reusing controller functions).
-   **Action**: All new client developments should use `/api/v1` prefixes.
