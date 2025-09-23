```mermaid
flowchart LR
    subgraph Clients
        C[Clients / Admin UI]
    end

    subgraph Edge
        G[API Gateway]
        A[Auth Service]
    end

    subgraph Ingestion
        I[Ingestion APIs\n(Upload, Webhooks, Crawlers)]
        Q[(Event Bus / Queue)]
    end

    subgraph Processing
        O[Job Orchestrator & Scheduler]
        P[Face Pipeline\nDetect → Align → Embed → pHash → Moderate]
    end

    subgraph Storage
        V[(Vector DB\nEmbeddings)]
        M[(Metadata DB\nPostgres)]
        S[(Object Store\nS3/MinIO)]
    end

    subgraph Query
        SE[Search API + Results Aggregator]
        AL[Alerts / Exports]
        AD[Admin Console]
        PO[Policy Engine]
    end

    %% Flows
    C --> G --> A --> I --> Q --> O --> P
    P --> V
    P --> M
    P --> S
    SE --> V
    SE --> M
    SE --> S
    C --> SE
    SE --> AL
    SE --> AD
    PO --> SE
    AD --> PO
