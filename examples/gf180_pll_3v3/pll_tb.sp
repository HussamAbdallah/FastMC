* FastMC Testbench — GF180 PLL (3.3V, 8→48 MHz)
* -------------------------------------------------------
* Set the two paths below before running.
* PDK_ROOT  : directory containing design.ngspice and sm141064.ngspice
* NETLIST   : RC-extracted PLL netlist (TOP_pex_extracted.spice or equivalent)
* -------------------------------------------------------

* 1. PDK STATISTICAL MODELS
* Replace the paths below with your local PDK location.
.include "PDK_ROOT/models/ngspice/design.ngspice"
.lib "PDK_ROOT/models/ngspice/sm141064.ngspice" statistical
.lib "PDK_ROOT/models/ngspice/sm141064.ngspice" bjt_statistical
.lib "PDK_ROOT/models/ngspice/sm141064.ngspice" res_statistical
.lib "PDK_ROOT/models/ngspice/sm141064.ngspice" moscap_typical
.lib "PDK_ROOT/models/ngspice/sm141064.ngspice" mimcap_statistical
.lib "PDK_ROOT/models/ngspice/sm141064.ngspice" diode_typical

* 2. PDK TRANSLATION WRAPPERS (Magic extracted → GF180 subcircuit names)
.subckt cap_mim_2f0_m4m5_noshield top bot c_width=10u c_length=10u
XC1 top bot mim_2p0fF c_width=c_width c_length=c_length
.ends

.subckt nfet_06v0 D G S B w=1u l=1u ad=0 as=0 pd=0 ps=0
X1 D G S B nmos_6p0 w=w l=l ad=ad as=as pd=pd ps=ps
.ends

.subckt pfet_06v0 D G S B w=1u l=1u ad=0 as=0 pd=0 ps=0
X1 D G S B pmos_6p0 w=w l=l ad=ad as=as pd=pd ps=ps
.ends

.subckt nfet_03v3 D G S B w=1u l=1u ad=0 as=0 pd=0 ps=0
X1 D G S B nmos_3p3 w=w l=l ad=ad as=as pd=pd ps=ps
.ends

.subckt pfet_03v3 D G S B w=1u l=1u ad=0 as=0 pd=0 ps=0
X1 D G S B pmos_3p3 w=w l=l ad=ad as=as pd=pd ps=ps
.ends

.subckt nfet_06v0_nvt D G S B w=1u l=1u ad=0 as=0 pd=0 ps=0
X1 D G S B nmos_6p0_nat w=w l=l ad=ad as=as pd=pd ps=ps
.ends

* 3. EXTRACTED NETLIST
* Replace with your RC-extracted PEX netlist path.
.include "NETLIST/TOP_pex_extracted.spice"

* 4. SIMULATION OPTIONS
.options rshunt=1e12 cshunt=1f method=gear reltol=1e-2 vntol=1m abstol=1u chgtol=1e-13

* 5. STIMULUS
* VDD ramps from 0→3.3V in 4ns; 8 MHz reference clock (125 ns period)
vpwr VDD 0 pulse(0 3.3 4n 0.8n 1p 10u 20u) dc 3.3
vext clk 0 pulse(0 3.3 4n 40p 40p 62.5n 125n) dc 0

* 6. DUT (pins: DEBUG VDD VSS GND OUT DISABLE REF_CLK VCTRL)
X1 nc_debug VDD 0 0 out 0 clk vctrl TOP

* 7. TRANSIENT ANALYSIS WITH MONTE CARLO
* 2 µs window (PLL locks at ~1.5 µs); FastMC controls the sweep count.
.tran 0.25n 2u sweep monte=100

* 8. MEASUREMENT — output clock period (target: ~20.83 ns → 48 MHz)
.measure tran t_period TRIG v(out) VAL=1.65 TD=1.5u RISE=1 \
                        TARG v(out) VAL=1.65 TD=1.5u RISE=2

.end
