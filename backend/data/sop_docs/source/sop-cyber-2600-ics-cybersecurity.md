# SOP-CYBER-2600: Industrial Control System Cybersecurity

**Document owner:** OT Security Department
**Effective date:** 2024-07-01
**Review cycle:** 12 months

## Section 1. Purpose and Scope

Governs cybersecurity controls for the site's industrial control systems (ICS), SCADA, and DCS networks — a separate, air-gapped-by-design network segment from corporate IT.

## Section 2. Network Segmentation

Clause 2.1 — The OT network shall remain physically or logically air-gapped from the corporate IT network and the public internet; any proposed data crossing (a historian feed, remote diagnostics link) requires OT Security Department review and a documented, monitored one-way or firewalled path — never an open bidirectional connection.

Clause 2.2 — A vendor remote-access request for diagnostics or support requires a time-boxed, supervised session through an approved jump host; standing/persistent remote access into the OT network is prohibited.

## Section 3. Removable Media

Clause 3.1 — USB drives and other removable media shall be scanned on a dedicated, air-gapped scanning station before connection to any OT asset; a media device shall never be connected directly to an OT asset without this scan, regardless of its source or apparent trustworthiness.

## Section 4. Patch Management

Clause 4.1 — OT asset patching follows a vendor-validated patch schedule, not the corporate IT cadence — an unvalidated patch can destabilize control logic. Patches are tested on an offline representative system before deployment to production OT assets.

## Section 5. Change Control

Clause 5.1 — Any change to ICS/SCADA configuration, logic, or network architecture requires MOC per SOP-MOC-2400 in addition to OT Security Department sign-off.

## Section 6. Incident Response

Clause 6.1 — A suspected OT security incident (unexpected behavior, unauthorized access indication) is reported to the OT Security Department immediately; the affected asset shall not be power-cycled or have logs cleared before forensic capture, unless continued operation poses an immediate safety hazard.

## Section 7. Records

Clause 7.1 — Network diagrams, patch records, and incident reports are filed to the OT Security File and retained for 5 years.
