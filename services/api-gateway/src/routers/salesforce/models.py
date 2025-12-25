"""
Salesforce Pydantic Models

Request and response schemas for Salesforce API operations.
Organized by object type for clarity.
"""

from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Base Models
# =============================================================================


class SalesforceRecord(BaseModel):
    """Base model for Salesforce records."""

    Id: str | None = None

    class Config:
        extra = "allow"


class SuccessResponse(BaseModel):
    """Standard success response."""

    success: bool = True


class RecordResponse(SuccessResponse):
    """Response containing a single record."""

    record: dict[str, Any]


class RecordsResponse(SuccessResponse):
    """Response containing multiple records."""

    records: list[dict[str, Any]]
    total: int


class CreateResponse(SuccessResponse):
    """Response for record creation."""

    id: str
    created: bool = True


class UpdateResponse(SuccessResponse):
    """Response for record update."""

    id: str
    updated: bool = True


class DeleteResponse(SuccessResponse):
    """Response for record deletion."""

    id: str
    deleted: bool = True


# =============================================================================
# Account Models
# =============================================================================


class AccountCreate(BaseModel):
    """Create a new Account."""

    Name: str = Field(..., min_length=1, max_length=255, description="Account name")
    Industry: str | None = Field(None, description="Industry sector")
    Phone: str | None = Field(None, description="Primary phone number")
    Website: str | None = Field(None, description="Company website URL")
    BillingCity: str | None = None
    BillingState: str | None = None
    BillingCountry: str | None = None
    Description: str | None = None
    AnnualRevenue: float | None = None
    NumberOfEmployees: int | None = None


class AccountUpdate(BaseModel):
    """Update an existing Account. All fields optional."""

    Name: str | None = Field(None, max_length=255)
    Industry: str | None = None
    Phone: str | None = None
    Website: str | None = None
    BillingCity: str | None = None
    BillingState: str | None = None
    Description: str | None = None


# =============================================================================
# Contact Models
# =============================================================================


class ContactCreate(BaseModel):
    """Create a new Contact."""

    FirstName: str | None = Field(None, max_length=40)
    LastName: str = Field(..., min_length=1, max_length=80)
    Email: str | None = Field(None, description="Email address")
    Phone: str | None = None
    Title: str | None = None
    AccountId: str | None = Field(None, description="Related Account ID")
    Department: str | None = None
    MailingCity: str | None = None
    MailingState: str | None = None


class ContactUpdate(BaseModel):
    """Update an existing Contact."""

    FirstName: str | None = None
    LastName: str | None = None
    Email: str | None = None
    Phone: str | None = None
    Title: str | None = None
    Department: str | None = None


# =============================================================================
# Opportunity Models
# =============================================================================


class OpportunityCreate(BaseModel):
    """Create a new Opportunity."""

    Name: str = Field(..., min_length=1, max_length=120)
    StageName: str = Field(default="Prospecting", description="Sales stage")
    CloseDate: str = Field(..., description="Expected close date (YYYY-MM-DD)")
    Amount: float | None = Field(None, ge=0, description="Deal value")
    AccountId: str | None = None
    Description: str | None = None
    Probability: float | None = Field(None, ge=0, le=100)
    LeadSource: str | None = None


class OpportunityUpdate(BaseModel):
    """Update an existing Opportunity."""

    Name: str | None = None
    StageName: str | None = None
    CloseDate: str | None = None
    Amount: float | None = None
    Description: str | None = None
    Probability: float | None = None


# =============================================================================
# Lead Models
# =============================================================================


class LeadCreate(BaseModel):
    """Create a new Lead."""

    FirstName: str | None = None
    LastName: str = Field(..., min_length=1)
    Company: str = Field(..., min_length=1, description="Lead's company name")
    Email: str | None = None
    Phone: str | None = None
    Title: str | None = None
    Status: str = Field(default="Open - Not Contacted")
    LeadSource: str | None = None
    Industry: str | None = None


class LeadUpdate(BaseModel):
    """Update an existing Lead."""

    FirstName: str | None = None
    LastName: str | None = None
    Company: str | None = None
    Email: str | None = None
    Phone: str | None = None
    Status: str | None = None


# =============================================================================
# Case Models
# =============================================================================


class CaseCreate(BaseModel):
    """Create a new Case (support ticket)."""

    Subject: str = Field(..., min_length=1, max_length=255)
    Description: str | None = None
    Status: str = Field(default="New")
    Priority: str = Field(default="Medium")
    Origin: str = Field(default="Web", description="How the case was submitted")
    AccountId: str | None = None
    ContactId: str | None = None
    Type: str | None = None


class CaseUpdate(BaseModel):
    """Update an existing Case."""

    Subject: str | None = None
    Description: str | None = None
    Status: str | None = None
    Priority: str | None = None


# =============================================================================
# Bulk API Models
# =============================================================================


class BulkQueryRequest(BaseModel):
    """Request to create a bulk query job."""

    query: str = Field(..., min_length=10, description="SOQL query")


class BulkIngestRequest(BaseModel):
    """Request to create a bulk ingest job."""

    operation: str = Field(..., pattern="^(insert|update|upsert|delete)$", description="Operation type")
    object_name: str = Field(..., min_length=1, description="Salesforce object name")
    external_id_field: str | None = Field(None, description="External ID field for upsert operations")


class BulkJobResponse(SuccessResponse):
    """Response for bulk job creation."""

    job_id: str
    state: str
    message: str | None = None


class BulkJobStatus(SuccessResponse):
    """Bulk job status response."""

    job_id: str
    state: str
    records_processed: int = 0
    records_failed: int = 0


# =============================================================================
# Composite API Models
# =============================================================================


class CompositeSubrequest(BaseModel):
    """Single subrequest in a composite call."""

    method: str = Field(..., pattern="^(GET|POST|PATCH|DELETE)$")
    url: str
    referenceId: str = Field(..., description="Unique identifier for this subrequest")
    body: dict[str, Any] | None = None


class CompositeRequest(BaseModel):
    """Composite API request with multiple subrequests."""

    allOrNone: bool = Field(default=True, description="If true, all subrequests must succeed or all fail")
    subrequests: list[CompositeSubrequest] = Field(..., max_length=25, description="List of subrequests (max 25)")


# =============================================================================
# SOQL Query Model
# =============================================================================


class SOQLRequest(BaseModel):
    """SOQL query request."""

    query: str = Field(..., min_length=10, description="SOQL query (SELECT only)")
